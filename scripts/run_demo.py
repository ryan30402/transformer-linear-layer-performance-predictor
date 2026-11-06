#!/usr/bin/env python3
"""Exercise the current offline workflow on explicitly simulated CPU data."""
from pathlib import Path
import argparse
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from linear_perf.experiment import run_experiment, audit_dataset
from linear_perf.workloads import build_workload_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plan = build_workload_manifest(profile='smoke')
    result = run_experiment(plan, args.out, session_id='cpu_smoke', simulate=True)
    audit = audit_dataset(args.out)
    if not audit['passed'] or result['status'] != 'simulated_complete':
        return 2
    print('SIMULATED software check completed. No GPU was measured.')
    print('Report:', Path(args.out).resolve() / 'analysis/report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
