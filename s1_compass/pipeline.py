"""Run partitioning, fine-tuning, calibration, evaluation, and a serving check."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from s1_compass.data.dataset import partition, batches
from s1_compass.data.tokenize import load_tokenizer
from s1_compass.model.backbone import mx
from s1_compass.model.scorer import probabilities
from s1_compass.train.loop import fit
from s1_compass.calibrate.temperature import fit as fit_temperature
from s1_compass.eval.metrics import evaluate
from s1_compass.eval.benchmark import run as evaluate_gates, benchmark_url, reliability_svg
from s1_compass.serve.api import post_decide, load_release, create_server
import threading
import argparse


def _predict(model, path, tokenizer, architecture="encoder"):
    logits_all=[]; probs_all=[]; labels=[]; options=[]
    question_ids=[]
    row_qids={row["id"]:row["question"]["id"] for row in (json.loads(line) for line in Path(path).read_text().splitlines())}
    sorted_qids=[row_qids[row_id] for row_id in sorted(row_qids)]
    from s1_compass.train.loop import _logits
    for batch in batches(path,1,17,tokenizer=tokenizer,architecture=architecture):
        z=_logits(model,batch); mx.eval(z)
        for i,label in enumerate(batch.label_idx):
            if label<0: continue
            n=len(batch.option_ids[i]); logits_all.append([float(v) for v in z[i,:n].tolist()]); labels.append(label); options.append(batch.option_ids[i])
            # batches() yields rows in the same deterministic ID order.
            question_ids.append(sorted_qids[len(question_ids)])
    return logits_all, labels, options, question_ids


def run(rows_path, annotations_path, snapshot_path, output_dir, *, base_model="modernbert-base", small_data=False, epochs=1, max_steps=None):
    out=Path(output_dir)
    from s1_compass.models.registry import get_base_model
    spec=get_base_model(base_model)
    out.mkdir(parents=True,exist_ok=True)
    snapshot_path=Path(snapshot_path) if snapshot_path else spec.snapshot_path
    tokenizer=load_tokenizer(snapshot_path)
    from s1_compass.models.snapshot import verify_snapshot
    lock=verify_snapshot(spec,snapshot_path)
    lock_sha=hashlib.sha256((Path(snapshot_path)/"s1-compass-snapshot.lock.json").read_bytes()).hexdigest()
    manifest=partition(rows_path,annotations_path,out/"splits",tokenizer=tokenizer,small_data=small_data,permutation_count=0,
                       base_model_id=spec.selector,preprocessing_version=spec.preprocessing_version,snapshot_lock_sha256=lock_sha,architecture=spec.architecture)
    training_mode="lora" if spec.architecture=="causal" else "full"
    trained=fit(manifest,snapshot_path,out/"candidate.npz",tokenizer=tokenizer,epochs=epochs,batch_size=1,max_steps=max_steps,base_model=base_model,training_mode=training_mode)
    predictions={}
    for split in ("calibration","test_normal"):
        predictions[split]=_predict(trained["model"],manifest.path.parent/f"{split}.jsonl",tokenizer,spec.architecture)
    tokenizer_hash=lock.get("tokenizer_sha256") or next(item["sha256"] for item in lock["files"] if item["path"]=="tokenizer.json")
    identity={"checkpoint_sha256":hashlib.sha256(Path(trained["path"]).read_bytes()).hexdigest(),
              "snapshot_lock_sha256":lock_sha,"tokenizer_sha256":tokenizer_hash,
              "preprocessing_hashes":manifest.data.get("preprocessing_hashes",{}),"base_model_id":spec.selector,
              "base_revision":lock["revision"],"architecture":spec.architecture,"hidden_size":spec.hidden_size,
              "preprocessing_version":spec.preprocessing_version}
    model_sha=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":" )).encode()).hexdigest()
    cal_logits,cal_labels,cal_ids,_=predictions["calibration"]
    calibration=fit_temperature(cal_logits,cal_labels,cal_ids,model_sha=model_sha,fit_on="calibration:"+hashlib.sha256(Path(manifest.path.parent/"calibration.jsonl").read_bytes()).hexdigest(),output_path=out/"calibration.json")
    test_logits,test_labels,test_ids,test_question_ids=predictions["test_normal"]
    calibrated=[]
    for row in test_logits:
        z=[x/calibration["T"] for x in row]; m=max(z); ex=[__import__('math').exp(v-m) for v in z]; s=sum(ex); calibrated.append([v/s for v in ex])
    evaluation=evaluate_gates(test_labels,calibrated,test_ids,question_ids=test_question_ids)
    evaluation["production_data_available"]=False
    evaluation["promotion_blocked_reason"]="Synthetic smoke rows are not eligible for release promotion."
    evaluation["promotable"]=False
    evaluation["checks"]["production_data_available"]=False
    evaluation["checks"]["full_acceptance_benchmark"]=False
    evaluation["training"]={k:v for k,v in trained.items() if k!="model"}
    evaluation["calibration"]={k:v for k,v in calibration.items() if k!="version"}
    evaluation["calibration_version"]=calibration["version"]
    evaluation["checkpoint_sha256"]=identity["checkpoint_sha256"]
    (out/"evaluation.json").write_text(json.dumps(evaluation,sort_keys=True,indent=2)+"\n")
    release={"format_version":1,"base_model_id":spec.selector,"base_revision":lock["revision"],"architecture":spec.architecture,"hidden_size":spec.hidden_size,"snapshot_lock_sha256":lock_sha,"tokenizer_sha256":tokenizer_hash,"preprocessing_version":spec.preprocessing_version,"checkpoint_sha256":identity["checkpoint_sha256"],"model_sha":model_sha,"calibration_version":calibration["version"],"evaluation_report_sha256":hashlib.sha256((out/"evaluation.json").read_bytes()).hexdigest(),"candidate_kind":"benchmark" if small_data else "candidate","training_mode":training_mode,"trainable_parameters":trained["trainable_parameters"]}
    (out/"release.json").write_text(json.dumps(release,sort_keys=True,indent=2)+"\n")
    # Validate the endpoint's schema/deadline path with a deterministic service stub.
    request={"state_text":"Smoke request", "questions":[{"id":"route","type":"choice","instructions":"Choose a route.","options":[{"id":"a","description":"Route A"},{"id":"b","description":"Route B"}]}]}
    service=load_release(out,snapshot_path,benchmark_mode=True,base_model=base_model)
    status,response=post_decide(service,request)
    evaluation["serving_smoke"]={"status":status,"schema_valid":status==200 or response.get("error",{}).get("code")=="LOW_CONFIDENCE","used_loaded_candidate":True}
    reliability_svg(calibrated,test_labels,path=out/"reliability.svg")
    server=create_server(service); threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        evaluation["loopback_http_smoke"]=benchmark_url(f"http://127.0.0.1:{server.server_port}/v1/decide",request,repeats=5)
    finally:
        server.shutdown(); server.server_close()
    (out/"evaluation.json").write_text(json.dumps(evaluation,sort_keys=True,indent=2)+"\n")
    release["evaluation_report_sha256"]=hashlib.sha256((out/"evaluation.json").read_bytes()).hexdigest()
    (out/"release.json").write_text(json.dumps(release,sort_keys=True,indent=2)+"\n")
    return evaluation


def main():
    parser=argparse.ArgumentParser(description="Train, calibrate, and evaluate an S1 Compass base-model candidate.")
    parser.add_argument("--base-model",choices=("modernbert-base","qwen3-0.6b"),default="modernbert-base")
    parser.add_argument("--snapshot",help="hash-locked local snapshot; defaults to the chosen base-model manifest")
    parser.add_argument("--rows",default="artifacts/smoke/rows.jsonl")
    parser.add_argument("--annotations",default="artifacts/smoke/annotations.json")
    parser.add_argument("--output",default=None)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--max-steps",type=int,default=None)
    parser.add_argument("--small-data",action="store_true")
    args=parser.parse_args()
    from s1_compass.models.registry import get_base_model
    spec=get_base_model(args.base_model)
    # Snapshots are immutable roots. Candidate outputs are siblings so training
    # artifacts can never alter the snapshot lock's file manifest.
    output=args.output or (Path("artifacts")/"runs"/spec.selector/"smoke")
    result=run(args.rows,args.annotations,args.snapshot or spec.snapshot_path,output,base_model=args.base_model,small_data=args.small_data,epochs=args.epochs,max_steps=args.max_steps)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == "__main__": main()
