"""Small-data capable MLX fine-tuning loop with Brier checkpoint selection."""
from __future__ import annotations
import hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any
from s1_compass.data.dataset import batches
from s1_compass.model.backbone import mx, nn, forward
from s1_compass.model.pooling import pool
from s1_compass.model.heads import initialize
from s1_compass.model.scorer import probabilities
from s1_compass.train.loss import cross_entropy


if nn is not None:
    class Model(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder,self.head=encoder,head
        def train(self): self.encoder.train(); self.head.train()
        def eval(self): self.encoder.eval(); self.head.eval()
else:
    class Model:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("MLX is unavailable; use the supported Apple Silicon runtime.")


def _flat_arrays(tree, prefix="", result=None):
    result={} if result is None else result
    if isinstance(tree,dict):
        for key,value in tree.items(): _flat_arrays(value,f"{prefix}.{key}" if prefix else str(key),result)
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree): _flat_arrays(value,f"{prefix}.{index}",result)
    else:
        result[prefix]=tree
    return result


def _scale(tree, factor):
    from mlx.utils import tree_map
    return tree_map(lambda value:value*factor,tree)


def _add(left, right):
    from mlx.utils import tree_map
    return tree_map(lambda a,b:a+b,left,right)


def _copy_tree(tree):
    if isinstance(tree,dict): return {k:_copy_tree(v) for k,v in tree.items()}
    if isinstance(tree,list): return [_copy_tree(v) for v in tree]
    if isinstance(tree,tuple): return tuple(_copy_tree(v) for v in tree)
    return tree


def _decay_matrices(module, learning_rate, weight_decay, trainable_only=False):
    def apply(tree, name=""):
        if isinstance(tree,dict): return {k:apply(v,f"{name}.{k}" if name else k) for k,v in tree.items()}
        if isinstance(tree,(list,tuple)): return type(tree)(apply(v,f"{name}.{i}") for i,v in enumerate(tree))
        if len(tree.shape)>=2 and not name.endswith("embeddings.weight"):
            return tree*(1.0-learning_rate*weight_decay)
        return tree
    parameters=module.trainable_parameters() if trainable_only else module.parameters()
    module.update(apply(parameters))


def _logits(model, batch, bf16=False):
    from s1_compass.model.heads import score
    return score(model.head,pool(forward(model.encoder,batch,bf16),batch),batch)


def fit(manifest, snapshot_path: str | Path, output_path: str | Path, *, tokenizer, epochs: int=10, batch_size: int=1, effective_batch_size: int=32, backbone_lr: float=2e-5, head_lr: float=1e-4, max_steps: int | None=None, base_model: str = "modernbert-base", training_mode: str = "full"):
    """Fine tune selected backbone and shared head; save best Brier state."""
    from s1_compass.model.backbone import load_base_model
    architecture = "causal" if base_model == "qwen3-0.6b" else "encoder"
    expected_mode = "lora" if architecture == "causal" else "full"
    if training_mode != expected_mode:
        raise ValueError(f"{base_model} requires explicit training_mode={expected_mode!r}.")
    from mlx import optimizers
    encoder=load_base_model(snapshot_path, base_model, require_parity=True, training_mode=training_mode)
    head=initialize(17, hidden_size=1024 if architecture == "causal" else 768)
    model=Model(encoder,head)
    backbone_optimizer=optimizers.AdamW(learning_rate=backbone_lr,weight_decay=0.0)
    head_optimizer=optimizers.AdamW(learning_rate=head_lr,weight_decay=0.0)
    def objective(model, batch):
        return cross_entropy(_logits(model,batch,bf16=True),batch)
    value_and_grad=nn.value_and_grad(model,objective)
    train_path=manifest.path.parent/"train.jsonl"
    val_path=manifest.path.parent/"validation.jsonl"
    planned=max(1,epochs*math.ceil(max(1,manifest.data.get("files",{}).get("train.jsonl",{}).get("rows",1))/effective_batch_size))
    warmup=max(1,math.ceil(.06*planned))
    best=float("inf"); best_parameters=None; best_optimizer_state=None; steps=0; stale=0; final_epoch=0
    for epoch in range(1,epochs+1):
        model.train()
        pending=None; pending_rows=0; losses=[]
        stream=batches(train_path,batch_size,17,tokenizer=tokenizer,training=True,epoch=epoch,architecture=architecture)
        def update_group():
            nonlocal pending,pending_rows,steps
            if pending is None: return
            grads,_=optimizers.clip_grad_norm(_scale(pending,1.0/pending_rows),1.0)
            steps+=1
            if steps<=warmup: multiplier=steps/warmup
            else:
                progress=min(1.0,(steps-warmup)/max(1,planned-warmup))
                multiplier=.1+.9*.5*(1+math.cos(math.pi*progress))
            encoder_lr=backbone_lr*multiplier; scalar_lr=head_lr*multiplier
            backbone_optimizer.learning_rate=encoder_lr; head_optimizer.learning_rate=scalar_lr
            backbone_optimizer.update(model.encoder,grads["encoder"])
            head_optimizer.update(model.head,grads["head"])
            _decay_matrices(model.encoder,encoder_lr,.01,trainable_only=architecture=="causal"); _decay_matrices(model.head,scalar_lr,.01)
            mx.eval(model.parameters(),backbone_optimizer.state,head_optimizer.state,*losses)
            if any(not math.isfinite(float(loss.item())) for loss in losses): raise FloatingPointError("nonfinite training loss")
            pending=None; pending_rows=0; losses.clear()
        for batch in stream:
            loss,grads=value_and_grad(model,batch)
            rows=len(batch.label_idx); mx.eval(loss,grads)
            pending=_add(pending,_scale(grads,rows)) if pending is not None else _scale(grads,rows)
            pending_rows+=rows; losses.append(loss)
            if pending_rows>=effective_batch_size: update_group()
            if max_steps is not None and steps>=max_steps: break
        update_group()
        model.eval()
        labels=[]; probs=[]; ids=[]
        for batch in batches(val_path,batch_size,17,tokenizer=tokenizer,architecture=architecture):
            p=probabilities(_logits(model,batch),1.0); mx.eval(p)
            for i,label in enumerate(batch.label_idx):
                if label<0: continue
                n=len(batch.option_ids[i]); labels.append(label); ids.append(batch.option_ids[i]); probs.append(p[i,:n].tolist())
        brier=sum(sum((float(v)-(j==y))**2 for j,v in enumerate(row)) for y,row in zip(labels,probs))/max(1,len(labels))
        if not math.isfinite(brier): raise FloatingPointError("nonfinite validation Brier")
        final_epoch=epoch
        if brier<best-.0001:
            current_parameters=model.trainable_parameters() if architecture=="causal" else model.parameters()
            best=brier; best_parameters=_copy_tree(current_parameters); best_optimizer_state={"backbone":_copy_tree(backbone_optimizer.state),"head":_copy_tree(head_optimizer.state)}; mx.eval(best_parameters,best_optimizer_state); stale=0
        else: stale+=1
        if (max_steps is not None and steps>=max_steps) or stale>=3: break
    if best_parameters is None: raise ValueError("validation split has no answerable rows")
    if architecture=="causal":
        model.update(best_parameters)
        mx.eval(model.trainable_parameters())
    else:
        model.update(best_parameters)
        mx.eval(model.parameters())
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+path.stem, suffix=".npz", dir=path.parent); os.close(fd)
    try:
        saved_parameters=model.trainable_parameters() if architecture=="causal" else model.parameters()
        mx.savez(tmp,**_flat_arrays(saved_parameters))
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    state_path=path.with_name(path.stem+"-training-state.npz")
    if best_optimizer_state is not None:
        fd,state_tmp=tempfile.mkstemp(prefix="."+state_path.stem,suffix=".npz",dir=path.parent); os.close(fd)
        try:
            mx.savez(state_tmp,**{f"{group}.{key}":value for group,tree in best_optimizer_state.items() for key,value in _flat_arrays(tree).items()})
            os.replace(state_tmp,state_path)
        finally:
            if os.path.exists(state_tmp): os.unlink(state_tmp)
    trainable_description=("LoRA rank 8, alpha 16 on q_proj and v_proj in final 4 Qwen3 layers + scalar head" if architecture=="causal" else "all ModernBERT backbone parameters + shared scalar head")
    metadata={"seed":17,"epochs":final_epoch,"steps":steps,"best_validation_brier":best,"manifest_sha256":hashlib.sha256(manifest.path.read_bytes()).hexdigest(),"backbone_lock":str(Path(snapshot_path)/("s1-compass-snapshot.lock.json" if architecture=="causal" else "backbone.lock.json")),"effective_batch_size":effective_batch_size,"microbatch_size":batch_size,"base_model_id":base_model,"training_mode":training_mode,"trainable_parameters":trainable_description,"architecture":architecture,"hidden_size":1024 if architecture=="causal" else 768}
    path.with_suffix(".json").write_text(json.dumps(metadata,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    return {"path":str(path),"best_validation_brier":best,"steps":steps,"epoch":final_epoch,"model":model,"training_state":str(state_path),"trainable_parameters":trainable_description}
