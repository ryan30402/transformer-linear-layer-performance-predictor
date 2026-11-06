"""Transformer Linear Layer Performance Predictor: dataset-first experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .common import LabError, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Build a latency-free workload manifest from model configurations")
    plan.add_argument("--profile", choices=("standard", "smoke"), default="standard")
    plan.add_argument("--seed", type=int, default=20261106)
    plan.add_argument("--out", required=True)
    for name in ("run", "collect", "experiment"):
        help_text = ("Run a fresh complete experiment: primary, repeat, comparison and report" if name == "experiment" else
                     "Collect every split before any fitting" + (" and analyze" if name == "run" else ""))
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=str(ROOT / "configs" / "workloads.json"))
        p.add_argument("--out", required=True)
        p.add_argument("--session-id", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--dtypes", nargs="+", choices=("float16", "bfloat16"))
        p.add_argument("--simulate", action="store_true", help="Explicit CPU software check, not GPU data")
        p.add_argument("--skip-profiles", action="store_true")
        p.add_argument("--l2-cache-mib", type=float, help="Explicit L2 size only when runtime discovery is unavailable")
        p.add_argument("--max-wall-seconds", type=int)
        p.add_argument("--checkpoint-directory", help="Publish verified evidence snapshots outside the local run directory")
        p.add_argument("--checkpoint-source-sha256", help="Expected checksum of the source archive supplied by the launcher")
        if name == "experiment":
            p.add_argument("--skip-repeat", action="store_true", help="Run a deliberately primary-only experiment")
            p.add_argument("--include-layer", action="store_true", help="Also run the optional measured decoder-layer check")
            p.add_argument("--layer-model-id", default="qwen2_5_0_5b")
            p.add_argument("--layer-dtype", choices=("float16", "bfloat16"))
        else:
            p.add_argument("--repeat-subset", action="store_true", help="Collect a shape-only 12.5 percent repeat subset; no model refit")
    p = sub.add_parser("analyze", help="Group-aware CV, refit and held-out evaluation of an existing dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("audit", help="Validate an existing dataset without GPU work or fitting")
    p.add_argument("--run", required=True)
    p = sub.add_parser("compare", help="Compare repeatability and transfer the source model without retraining")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("repeat", "gpu"), default="repeat")
    p = sub.add_parser("layer", help="Validate a hybrid random-weight decoder estimate on the source GPU")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default="qwen2_5_0_5b")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("doctor", help="Inspect PyTorch/CUDA availability")
    p = sub.add_parser("legacy", help="Explicit compatibility commands for the first-release native protocol")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "legacy":
            from .campaign import main as legacy_main
            return legacy_main(args.arguments)
        if args.command == "plan":
            from .workloads import build_workload_manifest
            result = build_workload_manifest(profile=args.profile, seed=args.seed)
            write_json(args.out, result, overwrite=False)
            result = {"status": "plan_created", "output": str(Path(args.out).resolve()), "cases": len(result["cases"])}
        elif args.command in {"run", "collect", "experiment"}:
            from .experiment import run_experiment
            from .dataset_measurement import repeat_subset
            plan = copy.deepcopy(read_json(args.config))
            if args.dtypes:
                plan["dtypes"] = args.dtypes
            settings = plan.setdefault("measurement", {})
            if args.skip_profiles:
                settings["profile"] = False
            if args.l2_cache_mib is not None:
                if not math.isfinite(args.l2_cache_mib) or args.l2_cache_mib <= 0:
                    raise LabError("L2 size must be positive")
                settings["l2_cache_bytes"] = int(args.l2_cache_mib * 1024**2)
            if args.max_wall_seconds is not None:
                settings["max_wall_seconds"] = args.max_wall_seconds
            if args.checkpoint_source_sha256 and not args.checkpoint_directory:
                raise LabError("A checkpoint source checksum requires --checkpoint-directory")
            checkpoints = None
            if args.checkpoint_directory:
                from .checkpoints import DurableCheckpoints
                checkpoints = DurableCheckpoints(args.out, args.checkpoint_directory,
                                                 session_id=args.session_id,
                                                 source_sha256=args.checkpoint_source_sha256)
            if args.command == "experiment":
                from .fresh_experiment import run_fresh_experiment
                result = run_fresh_experiment(plan, args.out, session_id=args.session_id, device=args.device,
                                              simulate=args.simulate, include_repeat=not args.skip_repeat,
                                              include_layer=args.include_layer, layer_model_id=args.layer_model_id,
                                              layer_dtype=args.layer_dtype,
                                              **({"checkpoint": checkpoints.stage,
                                                   "collection_checkpoint": checkpoints.collection,
                                                   "checkpoint_required": True} if checkpoints is not None else {}))
            else:
                ids = repeat_subset(plan) if args.repeat_subset else None
                result = run_experiment(plan, args.out, session_id=args.session_id, device=args.device,
                                        simulate=args.simulate, analyze=args.command == "run", repeat_ids=ids,
                                        **({"collection_checkpoint": checkpoints.collection,
                                             "checkpoint_required": True} if checkpoints is not None else {}))
        elif args.command == "analyze":
            from .experiment import analyze_dataset
            result = analyze_dataset(args.dataset, args.out)
        elif args.command == "audit":
            from .experiment import audit_dataset
            result = audit_dataset(args.run)
        elif args.command == "compare":
            from .experiment import compare_datasets
            result = compare_datasets(args.source, args.target, args.out, mode=args.mode)
        elif args.command == "layer":
            from .experiment import validate_layer
            result = validate_layer(args.source, args.out, model_id=args.model_id, dtype=args.dtype,
                                    batch_size=args.batch_size, sequence_length=args.sequence_length,
                                    device=args.device)
        else:
            from .benchmark import doctor
            result = doctor()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") in {"failed", "interrupted", "incomplete"} or result.get("passed") is False else 0
    except (LabError, OSError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
