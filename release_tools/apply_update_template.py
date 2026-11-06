"""Apply this reviewed project update to a clean local Git checkout.

Usage: python3 apply_predictor_update.py /path/to/checkout [--check-only]
The script creates a branch and applies the embedded patch. It does not commit,
push, install dependencies, execute downloaded code, or discard local changes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

BASE_COMMIT = "__BASE_COMMIT__"
PATCH_SHA256 = "__PATCH_SHA256__"
PATCH_B64 = "__PATCH_B64__"
EXPECTED_FILES_JSON = "__EXPECTED_FILES_JSON__"


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate without changing the checkout")
    parser.add_argument("--branch", default="improve-linear-performance-1-1")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_-]{0,100}", args.branch):
        parser.error("Use an alphanumeric branch name with slashes, underscores or dashes")
    repo = args.repo.expanduser().resolve()
    try:
        top = Path(git(repo, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
        if repo != top:
            raise ValueError("Pass the repository's top-level directory")
        origin = git(repo, "remote", "get-url", "origin").stdout.strip().removesuffix(".git").rstrip("/")
        if origin not in {"https://github.com/ryan30402/transformer-linear-layer-performance-predictor",
                          "git@github.com:ryan30402/transformer-linear-layer-performance-predictor"}:
            raise ValueError("The checkout origin is not the intended project")
        if git(repo, "status", "--porcelain", "--untracked-files=all").stdout.strip():
            raise ValueError("Checkout has local changes. Preserve them first, or use a fresh clone; nothing was modified")
        if git(repo, "rev-parse", "HEAD").stdout.strip() != BASE_COMMIT:
            print("The checkout is newer/different than the reviewed baseline. Checking patch applicability.")
        patch = base64.b64decode(PATCH_B64, validate=True)
        if hashlib.sha256(patch).hexdigest() != PATCH_SHA256:
            raise ValueError("Embedded patch checksum mismatch")
        expected = json.loads(EXPECTED_FILES_JSON)
        if git(repo, "show-ref", "--verify", "--quiet", "refs/heads/" + args.branch, check=False).returncode == 0:
            raise ValueError("Update branch already exists; use --branch with a new name")
        with tempfile.TemporaryDirectory(prefix="linear-perf-update-") as directory:
            patch_path = Path(directory) / "project.patch"
            patch_path.write_bytes(patch)
            git(repo, "apply", "--check", "--binary", str(patch_path))
            if args.check_only:
                print("Patch verified; checkout was not changed.")
                return 0
            git(repo, "switch", "-c", args.branch)
            git(repo, "apply", "--binary", str(patch_path))
        for name, checksum in expected.items():
            path = repo / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Applied file differs from the reviewed payload: " + name)
        print("Applied the update on branch " + args.branch + ". No commit or push was performed.")
        print(git(repo, "diff", "--stat").stdout)
        print("Review and publish using your own Git identity:")
        print("cd " + shlex.quote(str(repo)))
        print("git add -A")
        print('git commit -m "Improve offline validation and GPU measurement"')
        print("git push origin HEAD:main")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print("Update could not complete: " + str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stderr.strip())
        print("Existing work has not been deleted. Inspect git status before continuing.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
