# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The DeepAnalyze (DA) substrate: DA's HTTP agent API as an executor.

DA runs as one long-lived service (sessions + agent runs over HTTP); a rollout is
one DA session bound to the task workspace, driven from the host. The editable
surface maps onto DA's own mechanisms:

    AGENTS.md        c1..c4 always-on text -> DA's two-level project-instruction
                     loader (workspace AGENTS.md is injected at the system tail;
                     an absent file injects nothing, byte-identical to stock)
    skills/<n>/...   c6 -> DA agent_skills packages (loaded via the skill system)
    scripts/*.py     c7 -> packaged inside the skill tree

Sub-agents, plugins and hooks have no DA runtime equivalent and stay off the
surface. The trajectory is DA's transcript JSONL (one entry per line), rendered
into the shared [ACTION]/[OBSERVATION] grammar.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, RunRequest

DA_BASE_URL = os.environ.get("HARNESS2_DA_BASE_URL", "http://127.0.0.1:21000")
DA_MODE = os.environ.get("HARNESS2_DA_MODE", "code-it")

DA_SURFACE = SurfaceSpec(
    name="da",
    always_on=("AGENTS.md",),
    always_on_max=24_000,
    always_on_cap_reason="every token here competes with the agent's own output budget",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. "
        '"When <the situation this skill handles>, invoke the `{name}` skill."'
    ),
    autotrigger_file="AGENTS.md",
    trigger_line="- When {desc}, invoke the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "skills/README.md",
            "<!-- c6: reusable workflows loaded on demand through DA's skill system. "
            "Add one as skills/<name>/SKILL.md with frontmatter `name:` matching the "
            "directory and a `description:`. -->\n",
        ),
    ),
)

_DA_INSTALL_MODES = ("workspace_agents_md",)

_POLL_INTERVAL_S = 5.0


class DaError(RuntimeError):
    """A rollout-level failure talking to the DA service."""


@dataclass(frozen=True)
class DaRolloutResult:
    """One finished DA rollout: final status and the run's trajectory entries."""

    status: str
    output: str
    turns: int


class DaClient:
    """A minimal HTTP client for DA's session/agent API (stdlib only).

    The service owns the model wiring (its provider table points at whatever
    endpoint the operator configured), so this client never handles keys.
    """

    def __init__(self, base_url: str = DA_BASE_URL, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> object:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(
                req, timeout=timeout_s or self.timeout_s
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise DaError(f"{method} {path} -> {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise DaError(f"{method} {path} unreachable: {e.reason}") from e

    def health(self) -> dict:
        payload = self._request("GET", "/api/health")
        if not isinstance(payload, dict):
            raise DaError("unexpected /api/health payload")
        return payload

    def create_session(self, title: str, *, mode: str = DA_MODE) -> str:
        payload = self._request("POST", "/api/sessions", body={"title": title, "mode": mode})
        sid = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(sid, str) or not sid:
            raise DaError(f"create_session returned no id: {payload!r}")
        return sid

    def create_project(self, root_path: Path) -> str:
        """Register a host directory as a DA project; returns the project id.

        Host projects are the single sanctioned way to point a session's
        workspace at a benchmark task directory (requireHostProjects is on in
        local mode). Idempotent enough for retries: a duplicate rootPath is
        rejected by the server as a conflict, which callers may ignore if a
        lookup by rootPath already returned the id.
        """
        payload = self._request(
            "POST", "/api/projects", body={"rootPath": str(root_path.resolve())}
        )
        pid = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(pid, str) or not pid:
            raise DaError(f"create_project returned no id: {payload!r}")
        return pid

    def find_project(self, root_path: Path) -> str | None:
        payload = self._request("GET", "/api/projects")
        projects = payload if isinstance(payload, list) else (
            payload.get("projects") if isinstance(payload, dict) else []
        )
        wanted = str(root_path.resolve())
        if isinstance(projects, list):
            for p in projects:
                if isinstance(p, dict) and str(p.get("rootPath") or "") == wanted:
                    pid = p.get("id")
                    if isinstance(pid, str) and pid:
                        return pid
        return None

    def bind_workspace(self, session_id: str, project_id: str) -> None:
        self._request(
            "POST",
            f"/api/sessions/{session_id}/workspace-bind",
            body={"projectId": project_id},
        )

    def run_async(self, session_id: str, instruction: str) -> str:
        payload = self._request(
            "POST",
            "/api/agents/run-async",
            body={"sessionId": session_id, "input": instruction},
        )
        tid = payload.get("taskId") if isinstance(payload, dict) else None
        if not isinstance(tid, str) or not tid:
            raise DaError(f"run-async returned no taskId: {payload!r}")
        return tid

    def task_status(self, task_id: str) -> dict:
        payload = self._request("GET", f"/api/agents/tasks/{task_id}/status")
        if not isinstance(payload, dict):
            raise DaError(f"unexpected task status payload: {payload!r}")
        return payload

    def transcript(self, session_id: str) -> list[dict]:
        payload = self._request("GET", f"/api/sessions/{session_id}/transcript")
        if not isinstance(payload, dict):
            raise DaError("unexpected transcript payload")
        entries = payload.get("allEntries")
        if not isinstance(entries, list):
            raise DaError("transcript payload has no allEntries list")
        return [e for e in entries if isinstance(e, dict)]

    def wait_for_task(
        self, task_id: str, *, timeout_s: float, poll_interval_s: float = _POLL_INTERVAL_S
    ) -> DaRolloutResult:
        """Poll a run-async task to a terminal state (complete/error/cancelled)."""
        deadline = time.monotonic() + timeout_s
        last: dict = {}
        while True:
            last = self.task_status(task_id)
            state = last.get("state") or last.get("status") or ""
            if state in ("complete", "completed", "error", "failed", "cancelled"):
                out = last.get("output") if isinstance(last.get("output"), str) else ""
                turns = last.get("turnsUsed") if isinstance(last.get("turnsUsed"), int) else 0
                return DaRolloutResult(status=state, output=out, turns=turns)
            if time.monotonic() >= deadline:
                raise DaError(
                    f"task {task_id} did not reach a terminal state within "
                    f"{timeout_s:.0f}s (last state: {state!r})"
                )
            time.sleep(poll_interval_s)

    def delete_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/api/sessions/{session_id}")
        except DaError:
            pass


def render_events(
    path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
) -> str:
    """Render a DA transcript JSONL (one entry per line) into the shared grammar.

    Entry types: user / assistant (streamed text fragments) / tool_use /
    tool_result / turn_usage. Fragments of one assistant message are
    concatenated in order; only user, assistant, tool_use and tool_result are
    rendered.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(failed to read trajectory: {e})"
    lines: list[str] = []
    pending_assistant: list[str] = []

    def flush_assistant() -> None:
        if pending_assistant:
            text = "".join(pending_assistant).strip()
            del pending_assistant[:]
            if text:
                lines.append(f"[ASSISTANT] {_clip(text, obs_trunc)}")

    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et == "user":
            flush_assistant()
            content = ev.get("content")
            if isinstance(content, str) and content.strip():
                lines.append(f"[USER] {_clip(content, obs_trunc)}")
        elif et == "assistant":
            content = ev.get("content")
            if isinstance(content, str) and content:
                pending_assistant.append(content)
        elif et == "tool_use":
            flush_assistant()
            name = ev.get("toolName") or "tool"
            tin = ev.get("input")
            args = json.dumps(tin, default=str, ensure_ascii=False) if tin is not None else ""
            lines.append(f"[ACTION] {name}({_clip(args, args_trunc)})")
        elif et == "tool_result":
            flush_assistant()
            name = ev.get("toolName") or "tool"
            out = ev.get("output")
            is_error = bool(ev.get("isError"))
            if is_error:
                lines.append(f"[OBSERVATION ERROR] {_clip(out, obs_trunc)}")
            elif out is not None:
                lines.append(f"[OBSERVATION] {_clip(out, obs_trunc)}")
    flush_assistant()
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


def _clip(value: object, cap: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


_TRUNC_FMT = "… [trajectory truncated, {n} more characters dropped]"


class DaSubstrate:
    """The DA executor. A benchmark adapter supplies the launcher body, exactly
    like OpencodeSubstrate: the benchmark owns workspace preparation, run_dir
    layout and read-back locations; this class owns the DA-specific surface."""

    def __init__(self, launcher=None) -> None:
        self._launcher = launcher
        self.name = "da"

    def surface(self) -> SurfaceSpec:
        return DA_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        mode = getattr(spec, "install_mode", "workspace_agents_md")
        if mode not in _DA_INSTALL_MODES:
            raise ValueError(
                f"install_mode {mode!r} is not a da mode; this substrate supports "
                f"{_DA_INSTALL_MODES}"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(DA_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a da harness tree (no AGENTS.md present)")
        staged = Path(tempfile.mkdtemp(prefix="adapt-da-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in ctx.items():
                if key.startswith("file:"):
                    rel = HT.normalize_rel(key[len("file:") :])
                    p = staged / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
            fingerprint = RM.fingerprint(tree_dir)
            return Install(staging_dir=staged, fingerprint=fingerprint, extras={"install_mode": mode})
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "DaSubstrate was constructed without a launcher body; the benchmark "
                "adapter must supply one (it owns workspace prep, env wiring and "
                "read-back locations)"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Check the DA service is alive and its model wiring is healthy."""
        client = DaClient()
        try:
            health = client.health()
        except DaError as e:
            raise RuntimeError(
                f"DeepAnalyze service unreachable at {client.base_url}: {e}; "
                "start it (npm run dev:full:local) or set HARNESS2_DA_BASE_URL."
            ) from e
        llm = health.get("llm")
        if not isinstance(llm, dict) or llm.get("status") != "ok":
            raise RuntimeError(
                f"DA LLM wiring not healthy: {json.dumps(llm)[:200]}; configure a "
                "provider (PUT /api/settings/providers/:id) before running."
            )


__all__ = [
    "DA_SURFACE",
    "DA_BASE_URL",
    "DA_MODE",
    "DaClient",
    "DaError",
    "DaRolloutResult",
    "DaSubstrate",
    "render_events",
]
