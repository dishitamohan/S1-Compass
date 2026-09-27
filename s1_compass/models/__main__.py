"""Manage local, hash-locked model snapshots."""
from __future__ import annotations

import argparse

from s1_compass.models.registry import BASE_MODELS, get_base_model
from s1_compass.models.snapshot import download_snapshot, lock_existing_snapshot, verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m s1_compass.models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list registered base models and local snapshot status")
    subparsers.add_parser("parity", help="compare Qwen3 MLX hidden states with the pinned Transformers reference")
    download = subparsers.add_parser("download", help="download a base model at a pinned commit")
    download.add_argument("base_model", choices=tuple(BASE_MODELS))
    download.add_argument("--revision", required=True, help="immutable 40-character repository commit")
    download.add_argument("--destination", help="optional empty destination directory")
    verify = subparsers.add_parser("verify", help="verify a local snapshot against its hash lock")
    verify.add_argument("base_model", choices=tuple(BASE_MODELS))
    verify.add_argument("--snapshot", help="snapshot directory; defaults to the registry path")
    lock = subparsers.add_parser("lock", help="hash-lock an already downloaded snapshot without downloading it")
    lock.add_argument("base_model", choices=tuple(BASE_MODELS))
    lock.add_argument("--revision", required=True, help="immutable 40-character repository commit")
    lock.add_argument("--snapshot", help="snapshot directory; defaults to the registry path")
    args = parser.parse_args()

    if args.command == "parity":
        from s1_compass.models.parity import verify_qwen3
        result=verify_qwen3(get_base_model("qwen3-0.6b").snapshot_path)
        print(__import__("json").dumps(result,sort_keys=True,indent=2))
        if result["status"]!="passed": raise SystemExit(1)
    elif args.command == "list":
        for selector, spec in BASE_MODELS.items():
            if (spec.snapshot_path / "s1-compass-snapshot.lock.json").is_file():
                state = "hash-locked snapshot"
            elif (spec.snapshot_path / "backbone.lock.json").is_file() and spec.architecture == "encoder":
                state = "legacy pinned snapshot"
            elif (spec.snapshot_path / "config.json").is_file():
                state = "snapshot present, not hash-locked"
            else:
                state = "snapshot not installed"
            print(f"{selector}\t{spec.display_name}\t{spec.architecture}\t{state}\t{spec.snapshot_path}")
    elif args.command == "download":
        spec = get_base_model(args.base_model)
        path = download_snapshot(spec, args.revision, args.destination)
        print(f"Downloaded and locked {spec.model_id} at {args.revision}: {path}")
    elif args.command == "verify":
        spec = get_base_model(args.base_model)
        from pathlib import Path
        lock = verify_snapshot(spec, Path(args.snapshot) if args.snapshot else None)
        print(f"Verified {spec.model_id} at {lock['revision']} ({lock['files_sha256']})")
    else:
        spec = get_base_model(args.base_model)
        from pathlib import Path
        path = lock_existing_snapshot(spec, args.revision, Path(args.snapshot) if args.snapshot else None)
        print(f"Created immutable snapshot lock for {spec.model_id}: {path}")


if __name__ == "__main__":
    main()
