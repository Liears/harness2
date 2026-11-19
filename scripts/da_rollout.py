#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive one WorkBuddy-Bench code task through DeepAnalyze (DA) as executor.

Baseline smoke driver (no Harness2 recursion yet): for one task it prepares a
host workspace from the task's ``environment/workspace.tar.gz``, runs DA's
code-it mode against it, and assembles a run directory in the WorkBuddy layout
the verifier consumes:

    <run_dir>/
      agent/da-output.jsonl     trajectory (DA transcript entries, one per line)
      verifier/agent.patch      git diff of the workspace (the deliverable)

Scoring stays with the task's own verifier (tests/ + environment/), run
separately once this driver has produced the run_dir.

Usage:
    python3 scripts/da_rollout.py <task-id> [--dataset DIR] [--out DIR] [--timeout 1800]

Environment:
    HARNESS2_DA_BASE_URL   DA service base URL (default http://127.0.0.1:21000)
    HARNESS2_DA_MODE       DA session mode (default code-it)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness2.substrate.da import DA_DATA_DIR, DA_MODE, DaClient, DaError  # noqa: E402

WB_REPO = Path(__file__).resolve().parents[1] / "dependencies" / "workbuddy-bench"
DEFAULT_DATASET = WB_REPO / "datasets" / "wb-bench-code-v1.0" / "tasks"


def _log(msg: str) -> None:
    print(f"[da-rollout] {msg}", flush=True)


def prepare_workspace(task_dir: Path, dest: Path) -> None:
    """Unpack the task repo into dest; leak globs (tests/solution/environment)
    are never unpacked into the agent's workspace."""
    archive = task_dir / "environment" / "workspace.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"no workspace archive: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest, filter="data")


def workspace_diff(workspace: Path) -> str:
    """git diff of the agent's work, including untracked files.

    The upstream verifier scores a diff against the pristine repo, so the
    patch must carry both modifications and new files. The baseline commit is
    made by ``snapshot_baseline`` BEFORE the agent runs; this function only
    stages and diffs afterwards.
    """
    git = ["git", "-C", str(workspace)]
    status = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(git + ["add", "-A"], check=True)
    diff = subprocess.run(git + ["diff", "HEAD"], capture_output=True, text=True)
    return diff.stdout


def snapshot_baseline(workspace: Path) -> None:
    """Commit the pristine workspace so the agent's edits show up as a diff."""
    git = ["git", "-C", str(workspace)]
    base = subprocess.run(git + ["rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    if base.returncode != 0:
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(
            git + ["-c", "user.email=baseline@local", "-c", "user.name=baseline",
                   "commit", "-q", "-m", "baseline"],
            check=True,
        )


def run_one(task_id: str, *, dataset: Path, out_root: Path, timeout_s: int) -> Path:
    task_dir = dataset / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"unknown task: {task_dir}")
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()

    run_dir = out_root / task_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "agent").mkdir()
    (run_dir / "verifier").mkdir()

    # A stable per-task path: DA registers host directories as projects keyed
    # by a filesystem identity that collides for fresh mkdtemp paths (23505
    # on POST /api/projects), so we reuse one directory per task id instead.
    workspace = Path(tempfile.gettempdir()) / "da-wb-workspaces" / task_id
    if workspace.exists():
        shutil.rmtree(workspace)
    client = DaClient()
    session_id = ""
    try:
        _log(f"preparing workspace from {task_dir.name}")
        prepare_workspace(task_dir, workspace)
        snapshot_baseline(workspace)
        _log(f"workspace ready: {workspace}")

        project_id = client.find_project(workspace) or client.create_project(workspace)
        session_id = client.create_session(f"wb-code {task_id}")
        client.bind_workspace(session_id, project_id)
        _log(f"session {session_id} bound to project {project_id}")

        _log(f"dispatching task (timeout {timeout_s}s)")
        task_api_id = client.run_async(session_id, instruction)
        transcript_dir = (
            Path(DA_DATA_DIR) / "sessions" / session_id / "transcripts" if DA_DATA_DIR else None
        )
        result = client.wait_for_task(task_api_id, timeout_s=timeout_s, transcript_dir=transcript_dir)
        _log(f"task finished: status={result.status} turns={result.turns}")

        entries = client.transcript(session_id)
        stream = run_dir / "agent" / "da-output.jsonl"
        with stream.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _log(f"trajectory: {len(entries)} entries -> {stream}")

        patch = workspace_diff(workspace)
        (run_dir / "verifier" / "agent.patch").write_text(patch, encoding="utf-8")
        _log(f"agent.patch: {len(patch)} bytes")

        meta = {
            "task": task_id,
            "sessionId": session_id,
            "taskId": task_api_id,
            "status": result.status,
            "turns": result.turns,
            "output": result.output[:2000],
        }
        (run_dir / "da-meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.status not in ("complete", "completed"):
            _log(f"WARNING: non-complete status {result.status!r}; patch may be empty")
        return run_dir
    finally:
        # The session and its transcript stay on the DA side for traceability:
        # deleting them here cascades agent_tasks away and erases the on-disk
        # transcript mid-batch (observed: an interrupted model restart turned
        # four in-flight rollouts into errors, whose cleanup wiped the DB).
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_id")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=Path("runs/da-baseline"))
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args(argv)
    try:
        run_dir = run_one(
            args.task_id, dataset=args.dataset, out_root=args.out, timeout_s=args.timeout
        )
    except (FileNotFoundError, DaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
