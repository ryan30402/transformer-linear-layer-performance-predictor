#!/usr/bin/env python3
"""Run and audit a synthetic session on a CPU; no GPU performance is measured."""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from linear_perf.campaign import audit_run, run_campaign
from linear_perf.common import LabError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new directory; existing evidence is never overwritten")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.exists():
        parser.error("Output exists. Choose a fresh directory.")
    try:
        status = run_campaign(PROJECT / "configs" / (args.dtype + "_graph.json"), output,
                              session_id="SIMULATED_" + args.dtype, simulate=True)
        audit = audit_run(output)
        if status["status"] != "simulated_complete" or not audit["passed"]:
            print("Synthetic session did not complete its audit; inspect the retained output.", file=sys.stderr)
            return 2
    except (LabError, OSError, ValueError) as error:
        print(f"Failed: {error}", file=sys.stderr)
        return 2
    print(f"\nSIMULATED SOFTWARE VALIDATION ONLY. Report: {output / 'report.html'}")
    print("No GPU performance or numerical output was measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
