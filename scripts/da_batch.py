#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch baseline smoke: N WorkBuddy-Bench code tasks through DA, then verify.

Runs rollouts with a bounded thread pool, then judges each run dir and prints a
summary table. Every run_dir is rebuilt from scratch (each rollout gets a fresh
workspace), so a rerun re-executes the tasks rather than reusing old patches.
A rollout or judge failure is recorded as such and never laundered into a 0
score -- an unscored task stays unscored in the summary.

Usage:
    .venv/bin/python scripts/da_batch.py [task-id ...] [--n 10] [--jobs 4] \
        [--out DIR] [--timeout 1800]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.da_rollout import DEFAULT_DATASET, run_one  # noqa: E402

DEFAULT_OUT = Path("runs/da-baseline")


def pick_tasks(dataset: Path, n: int) -> list[str]:
    tasks = sorted(p.name for p in dataset.iterdir() if (p / "task.toml").is_file())
    return tasks[:n]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_ids", nargs="*")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument(
        "--harness", type=Path, default=None,
        help="harness tree (AGENTS.md, skills/) laid into every workspace before its run",
    )
    args = p.parse_args(argv)

    task_ids = args.task_ids or pick_tasks(args.dataset, args.n)
    print(f"[batch] {len(task_ids)} tasks, {args.jobs} parallel: {task_ids}", flush=True)

    results: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {
            ex.submit(run_one, tid, dataset=args.dataset, out_root=args.out,
                      timeout_s=args.timeout, harness=args.harness): tid
            for tid in task_ids
        }
        for fut in cf.as_completed(futs):
            tid = futs[fut]
            try:
                run_dir = fut.result()
                results[tid] = {"run_dir": str(run_dir), "ok": True}
                print(f"[batch] rollout done: {tid}", flush=True)
            except Exception as e:  # noqa: BLE001 - batch must survive one bad task
                # Infrastructure failures (queue timeout, provider timeout,
                # transport error) are NOT model failures: they mean "this
                # trial never happened", not "the agent scored 0". Mark them
                # unscored so the A/B comparison only covers real outcomes.
                msg = str(e)
                infra = any(
                    k in msg for k in (
                        "did not reach a terminal state",
                        "connection timeout",
                        "排队超时",
                        "ConnectionError",
                        "TimeoutError",
                    )
                )
                results[tid] = {
                    "ok": False,
                    "infra_failure": infra,
                    "error": msg[:300],
                }
                print(f"[batch] rollout {'INFRA-FAIL' if infra else 'FAILED'}: {tid}: {msg[:150]}", flush=True)

    from scripts.da_verify import verify_one

    for tid in task_ids:
        if not results[tid]["ok"]:
            continue
        try:
            score = verify_one(tid, run_dir=args.out / tid, dataset=args.dataset)
            results[tid]["score"] = score
            print(f"[batch] judge: {tid} -> {score}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[tid]["score"] = None
            results[tid]["judge_error"] = str(e)[:300]
            print(f"[batch] judge FAILED: {tid}: {e}", flush=True)

    scored = [r.get("score") for r in results.values() if r.get("score") is not None]
    infra = sum(1 for r in results.values() if r.get("infra_failure"))
    print("\n=== summary ===")
    for tid in task_ids:
        r = results[tid]
        mark = r.get("score")
        label = (
            "INFRA-FAIL" if r.get("infra_failure")
            else "FAILED" if not r["ok"]
            else "unscored" if mark is None
            else f"{mark:.3f}"
        )
        print(f"{tid:55s} {label}")
    if scored:
        print(f"\nmean={sum(scored) / len(scored):.3f}  scored={len(scored)}/{len(task_ids)}"
              + (f"  infra-failures={infra}" if infra else ""))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[batch] summary written: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
