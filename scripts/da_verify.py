#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score one DA-produced run_dir against a WorkBuddy-Bench code task.

Mirrors the dataset's host-side verifier contract (pytest_injected family):

    1. build the task's own ``environment/`` image (deps baked in)
    2. run one container: apply the agent patch to the pristine workspace,
       inject the task's tests, run the ``verifier.toml`` pytest command
    3. feed the JUnit XML + patches to the task's baked-in ``scorer.py``
    4. write ``<run_dir>/verifier/score.json`` (harness2 reads
       reward > overall > test_pass_rate > tests_passed/tests_total)

Usage:
    python3 scripts/da_verify.py <task-id> [--run-dir DIR] [--dataset DIR]

The run_dir is what scripts/da_rollout.py produced (verifier/agent.patch).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

WB_REPO = Path(__file__).resolve().parents[1] / "dependencies" / "workbuddy-bench"
DEFAULT_DATASET = WB_REPO / "datasets" / "wb-bench-code-v1.0" / "tasks"
# Direct pypi.org streams stall on this network; the same packages from a
# mirror keep the task's own Dockerfile semantics (only the index differs).
PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple"


def _log(msg: str) -> None:
    print(f"[da-verify] {msg}", flush=True)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    _log("$ " + shlex.join(cmd))
    return subprocess.run(cmd, **kwargs)


def build_context(task_dir: Path, build_dir: Path) -> Path:
    """Copy the task environment and point its pip installs at a mirror.

    The task's Dockerfile installs the repo's real dependencies; rewriting
    only the index URL changes where identical packages are downloaded from,
    not what gets installed or how the verifier behaves.
    """
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(task_dir / "environment", build_dir)
    dockerfile = build_dir / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    patched = re.sub(
        r"(python -m pip install|(?<![a-z])pip install)(?! -i\b)",
        rf"\1 -i {PIP_INDEX}",
        text,
    )
    if patched != text:
        dockerfile.write_text(patched, encoding="utf-8")
        _log("pip installs routed through mirror index (identical packages)")
    return build_dir


def verify_one(task_id: str, *, run_dir: Path, dataset: Path) -> float | None:
    task_dir = dataset / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"unknown task: {task_dir}")
    patch = run_dir / "verifier" / "agent.patch"
    if not patch.is_file():
        raise FileNotFoundError(f"missing agent.patch: {patch}")

    verifier_toml = tomllib.loads((task_dir / "tests" / "verifier.toml").read_text("utf-8"))
    family = verifier_toml.get("family", "pytest_injected")
    run_cfg = verifier_toml.get("run", {})
    pytest_cmd = run_cfg.get("command")
    cwd = run_cfg.get("cwd", "/workspace")
    if family == "pytest_injected" and (not isinstance(pytest_cmd, str) or not pytest_cmd.strip()):
        raise ValueError(f"{task_dir}/tests/verifier.toml has no run.command")
    if family not in ("pytest_injected", "script_verifier"):
        raise ValueError(f"{task_dir}/tests/verifier.toml: unsupported family {family!r}")

    image = f"da-verify/{task_id.lower().replace('_', '-')}"
    context = build_context(task_dir, run_dir / "build-context")
    build = _run(
        ["docker", "build", "-t", image, str(context)],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"docker build failed:\n{build.stderr[-2000:]}")
    _log(f"image built: {image}")

    score_json = run_dir / "verifier" / "score.json"
    score_json.parent.mkdir(parents=True, exist_ok=True)
    if family == "script_verifier":
        # The task ships its own tests/verifier.py: import the patched workspace
        # and write reward.json (same fields harness2's score reader expects)
        # into LOG_DIR. No injected tests, no baked scorer.
        script = f"""
set -uo pipefail
cd {shlex.quote(cwd)}
if ! git apply --whitespace=nowarn /tmp/agent.patch; then
  echo "patch application failed" > /logs/verifier/test_output.txt
  printf '{{"overall": 0.0, "test_status": "patch_failed", "tests_total": 1, "tests_passed": 0}}' > /logs/verifier/reward.json
  exit 0
fi
python /tmp/verifier.py
"""
        _run(
            [
                "docker", "run", "--rm",
                "-v", f"{(run_dir / 'verifier').resolve()}:/logs/verifier",
                "-v", f"{patch.resolve()}:/tmp/agent.patch:ro",
                "-v", f"{(task_dir / 'tests' / 'verifier.py').resolve()}:/tmp/verifier.py:ro",
                image, "bash", "-c", script,
            ],
        )
        reward = run_dir / "verifier" / "reward.json"
        if reward.is_file():
            shutil.copy(reward, score_json)
    else:
        script = f"""
set -uo pipefail
cd {shlex.quote(cwd)}
if ! git apply --whitespace=nowarn /tmp/agent.patch; then
  echo "patch application failed" > /logs/verifier/test_output.txt
  python /scorer/scorer.py --agent-patch /tmp/agent.patch --gold-patch /tmp/gold.patch \
    --test-status patch_failed --output /logs/verifier/score.json
  exit 0
fi
mkdir -p tests
cp -r /tmp/injected/. tests/
{pytest_cmd} || true
python /scorer/scorer.py --agent-patch /tmp/agent.patch --gold-patch /tmp/gold.patch \
  --junit-xml /logs/verifier/results.xml --output /logs/verifier/score.json
"""
        _run(
            [
                "docker", "run", "--rm",
                "-v", f"{(run_dir / 'verifier').resolve()}:/logs/verifier",
                "-v", f"{patch.resolve()}:/tmp/agent.patch:ro",
                "-v", f"{(task_dir / 'tests' / 'gold.patch').resolve()}:/tmp/gold.patch:ro",
                "-v", f"{(task_dir / 'tests' / 'injected' / 'tests').resolve()}:/tmp/injected:ro",
                image, "bash", "-c", script,
            ],
        )
    if not score_json.is_file():
        raise RuntimeError(f"verifier produced no score.json in {run_dir}")
    payload = json.loads(score_json.read_text("utf-8"))
    _log(f"score payload: {json.dumps(payload, ensure_ascii=False)[:400]}")
    for key in ("reward", "overall", "test_pass_rate"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
            return float(value)
    passed, total = payload.get("tests_passed"), payload.get("tests_total")
    if isinstance(passed, int) and isinstance(total, int) and total > 0:
        return max(0.0, min(passed / total, 1.0))
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_id")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="defaults to runs/da-baseline/<task_id>")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = p.parse_args(argv)
    run_dir = args.run_dir or (Path("runs/da-baseline") / args.task_id)
    try:
        score = verify_one(args.task_id, run_dir=run_dir, dataset=args.dataset)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"score: {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
