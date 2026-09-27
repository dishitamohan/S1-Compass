"""Reference parity gate for the locked Qwen3 MLX backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_cases(snapshot: Path, fixture_path: Path):
    from s1_compass.data.tokenize import encode_qwen, load_tokenizer
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    lock = json.loads((snapshot / "s1-compass-snapshot.lock.json").read_text(encoding="utf-8"))
    if fixture.get("revision") != lock["revision"] or fixture.get("tokenizer_sha256") != lock["tokenizer_sha256"]:
        raise ValueError("Qwen tokenizer golden fixture does not match the locked snapshot.")
    tokenizer = load_tokenizer(snapshot)
    result = []
    for name, case in sorted(fixture["fixtures"].items()):
        sequences, starts = encode_qwen(case["row"], tokenizer)
        if sequences != [ids[:length] for ids, length in zip(case["input_ids"], case["sequence_lengths"])]:
            raise ValueError(f"Qwen tokenizer golden mismatch in {name}.")
        result.append((name, sequences, starts, [o["id"] for o in case["row"]["options"]]))
    return result, fixture


def _reference_worker(snapshot: Path, fixture_path: Path, output_path: Path):
    import numpy as np
    import torch
    from transformers import AutoModel
    cases, _ = _fixture_cases(snapshot, fixture_path)
    # Qwen3 exposes last_hidden_state via Qwen3Model; explicitly load the base
    # model so the tied LM head is not materialized for this parity reference.
    from transformers import Qwen3Model
    model = Qwen3Model.from_pretrained(str(snapshot), local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    arrays = {}
    with torch.inference_mode():
        for index, (name, sequences, _, _) in enumerate(cases):
            width = max(map(len, sequences))
            pad_id = 151643
            ids = [seq + [pad_id] * (width - len(seq)) for seq in sequences]
            masks = [[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]
            output = model(input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=torch.tensor(masks, dtype=torch.long))
            arrays[f"hidden_{index}"] = output.last_hidden_state.float().cpu().numpy()
            arrays[f"lengths_{index}"] = np.asarray([len(seq) for seq in sequences], dtype=np.int32)
            arrays[f"starts_{index}"] = np.asarray(cases[index][2], dtype=np.int32)
    np.savez(output_path, **arrays)


BF16_MAX_ABS_ERROR = 2.0
BF16_MEAN_ABS_ERROR = 0.1
BF16_MEAN_COSINE = 0.9998


def verify_qwen3(snapshot_path: str | Path, *, tolerance: float = BF16_MAX_ABS_ERROR) -> dict:
    import numpy as np
    import mlx.core as mx
    from s1_compass.model.backbone import load_base_model
    from s1_compass.models.snapshot import verify_snapshot
    snapshot = Path(snapshot_path).resolve()
    lock = verify_snapshot(__import__("s1_compass.models.registry", fromlist=["get_base_model"]).get_base_model("qwen3-0.6b"), snapshot)
    fixture_path = Path(__file__).resolve().parents[1] / "data/fixtures/qwen3-tokenizer-golden.json"
    cases, fixture = _fixture_cases(snapshot, fixture_path)
    with tempfile.TemporaryDirectory(prefix="s1-qwen-parity-") as temporary:
        expected_path = Path(temporary) / "reference.npz"
        command = [sys.executable, "-m", "s1_compass.models.parity", "--reference-worker", str(snapshot), str(fixture_path), str(expected_path)]
        subprocess.run(command, check=True, env={**os.environ, "S1_COMPASS_DISABLE_MLX": "1"})
        reference = np.load(expected_path)
        encoder = load_base_model(snapshot, "qwen3-0.6b", require_parity=False, training_mode="lora")
        # Deterministic scalar probe makes parity include option ranking, not just tensor distance.
        projection = np.random.default_rng(91).normal(size=(int(lock["hidden_size"]),)).astype(np.float32)
        max_abs, total_abs, total_values, cosine_sum, cosine_count = 0.0, 0.0, 0, 0.0, 0
        decisions_match = decisions_total = 0
        actual_cases = {}
        for index, (name, sequences, starts, option_ids) in enumerate(cases):
            actual = encoder.forward_options([sequences], [starts])[0]
            mx.eval(actual)
            actual = np.asarray(actual.astype(mx.float32))
            actual_cases[name] = actual
            expected = reference[f"hidden_{index}"]
            lengths = reference[f"lengths_{index}"]
            start_values = reference[f"starts_{index}"]
            predicted_scores, reference_scores = [], []
            for option_index, length in enumerate(lengths.tolist()):
                left, right = int(start_values[option_index]), int(length)
                a = actual[option_index, left:right]
                b = expected[option_index, left:right]
                difference = np.abs(a - b)
                max_abs = max(max_abs, float(difference.max(initial=0.0)))
                total_abs += float(difference.sum()); total_values += difference.size
                av, bv = a.mean(axis=0), b.mean(axis=0)
                denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
                cosine_sum += float(np.dot(av, bv) / denom) if denom else 0.0
                cosine_count += 1
                predicted_scores.append(float(np.dot(av, projection)))
                reference_scores.append(float(np.dot(bv, projection)))
            if len(option_ids) > 1:
                decisions_match += int(max(range(len(option_ids)), key=lambda j: (predicted_scores[j], tuple(-ord(c) for c in option_ids[j]))) ==
                                        max(range(len(option_ids)), key=lambda j: (reference_scores[j], tuple(-ord(c) for c in option_ids[j]))))
                decisions_total += 1
        # Changing one option cannot alter another option's causal evidence.
        ordinary = next(case for case in cases if case[0] == "ordinary")
        from s1_compass.data.tokenize import load_tokenizer, _encode
        tokenizer = load_tokenizer(snapshot)
        altered = list(ordinary[1])
        start0 = ordinary[2][0]
        altered[0] = altered[0][:start0] + _encode(tokenizer, "A completely different replacement candidate response.")
        independent_hidden = encoder.forward_options([altered], [ordinary[2]])[0]
        mx.eval(independent_hidden)
        independent_hidden = np.asarray(independent_hidden.astype(mx.float32))
        first_other_len = len(ordinary[1][1])
        option_independence_error = float(np.max(np.abs(independent_hidden[1, :first_other_len] - actual_cases["ordinary"][1, :first_other_len])))
        # Reordering options changes only their slots; each corresponding representation remains stable.
        reordered = next(case for case in cases if case[0] == "reordered")
        permutation_errors=[]
        for option_id in ordinary[3]:
            oi=ordinary[3].index(option_id); ri=reordered[3].index(option_id)
            left=actual_cases["ordinary"][oi,ordinary[2][oi]:len(ordinary[1][oi])].mean(axis=0)
            right=actual_cases["reordered"][ri,reordered[2][ri]:len(reordered[1][ri])].mean(axis=0)
            permutation_errors.append(float(np.max(np.abs(left-right))))
        permutation_error=max(permutation_errors,default=0.0)
        # Tiny forward/backward probe confirms gradients reach LoRA and head weights.
        from s1_compass.data.tokenize import encode
        from s1_compass.model.heads import initialize
        from s1_compass.train.loop import Model, _logits
        from mlx import nn as mlx_nn
        train_row=dict(fixture["fixtures"]["ordinary"]["row"])
        train_row.update({"label":ordinary[3][0],"annotator_confidence":1.0,"source":"adjudicated"})
        probe_batch=encode([train_row],False,17,tokenizer=tokenizer,architecture="causal")
        probe=Model(encoder,initialize(91,1024))
        gradient_fn=mlx_nn.value_and_grad(probe,lambda current: mx.mean(_logits(current,probe_batch)))
        gradient_loss,gradient_tree=gradient_fn(probe)
        mx.eval(gradient_loss,gradient_tree)
        from mlx.utils import tree_flatten
        encoder_grads=tree_flatten(gradient_tree["encoder"])
        head_grads=tree_flatten(gradient_tree["head"])
        gradient_norm=sum(float(mx.sum(mx.abs(value)).item()) for _,value in encoder_grads)
        head_gradient_norm=sum(float(mx.sum(mx.abs(value)).item()) for _,value in head_grads)
        gradients_finite=all(bool(mx.all(mx.isfinite(value)).item()) for _,value in encoder_grads+head_grads)
        mean_abs = total_abs / max(total_values, 1)
        cosine = cosine_sum / max(cosine_count, 1)
        decision_agreement = decisions_match / max(decisions_total, 1)
        # The downloaded MLX-LM checkpoint is BF16 while the independent
        # Transformers oracle executes in FP32. Use tensor bounds calibrated
        # for that dtype comparison, then require near-identical pooled
        # directions and exact ranking agreement before permitting training.
        status = "passed" if max_abs <= tolerance and mean_abs <= BF16_MEAN_ABS_ERROR and cosine >= BF16_MEAN_COSINE and decision_agreement == 1.0 and option_independence_error <= 1e-5 and permutation_error <= 1e-4 and gradients_finite and gradient_norm > 0 and head_gradient_norm > 0 else "failed"
        result = {"format_version": 1, "status": status, "base_model_id": "qwen3-0.6b", "revision": lock["revision"],
                  "snapshot_lock_sha256": _sha256(snapshot / "s1-compass-snapshot.lock.json"),
                  "tokenizer_golden_sha256": _sha256(fixture_path), "mlx_lm_version": __import__("mlx_lm").__version__,
                  "transformers_version": __import__("transformers").__version__, "dtype": "BF16 MLX checkpoint; FP32 Transformers reference",
                  "cases": [case[0] for case in cases], "max_abs_error": max_abs, "mean_abs_error": mean_abs,
                  "mean_option_cosine_similarity": cosine, "decision_agreement": decision_agreement,
                  "option_independence_max_abs_error": option_independence_error,"permutation_mean_pool_max_abs_error":permutation_error,
                  "gradient_probe":{"finite":gradients_finite,"lora_l1_norm":gradient_norm,"head_l1_norm":head_gradient_norm},
                  "tolerances": {"max_abs_error": tolerance, "mean_abs_error": BF16_MEAN_ABS_ERROR, "mean_option_cosine_similarity": BF16_MEAN_COSINE, "decision_agreement": 1.0,"option_independence_max_abs_error":1e-5,"permutation_mean_pool_max_abs_error":1e-4}}
        result_path = snapshot.parent / (snapshot.name + ".parity.json")
        result_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-worker", nargs=3, metavar=("SNAPSHOT", "FIXTURE", "OUTPUT"))
    parser.add_argument("--snapshot", default="artifacts/qwen3-0.6b")
    args = parser.parse_args()
    if args.reference_worker:
        _reference_worker(*(Path(value) for value in args.reference_worker))
        return
    result = verify_qwen3(args.snapshot)
    print(json.dumps(result, sort_keys=True, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
