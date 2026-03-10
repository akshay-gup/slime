"""Bash tool sandbox for multi-turn RLVR environments."""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WORKDIR = "/opt/NeMo/slime_bash_tool_workspace"
DEFAULT_TRACE_DIR = "/opt/NeMo/slime_bash_tool_traces"
REWARD_RESULT_FILE = "solution.md"

EPHEMERAL_FILE_PATTERNS = [
    r"(?i)^(answer|answers|solution|solutions)(\.(md|txt|py))?$",
    r"(?i)^(final|new|temp|tmp)[_-]?(answer|solution)s?(\.(md|txt|py))?$",
    r"(?i)^(answer|solution)s?[_-](final|new|temp|tmp)(\.(md|txt|py))?$",
    r"(?i).*(tmp|temp|draft|copy|backup|bak)(\..*)?$",
    r"(?i).*[_-]v[0-9]+(\..*)?$",
    r"(?i).*[_-](ver|version)[0-9]+(\..*)?$",
    r"(?i)^(step|stage|test|trial|scratch|tmp)[0-9]+(\..*)?$",
    r"(?i)^(hello|script|run|temp|tmp|debug|check)(\..*)?$",
    r"(?i)^(problem|prompt|task|instruction|question)(\..*)?$",
    r"(?i).*(stdout|stderr|output|result|log)(\..*)?$",
    r"(?i)^test[_-]?.*",
]

EPHEMERAL_DIR_PATTERNS = [
    r"(?i)^(answer|answers|solution|solutions|tmp|temp|scratch|draft|backup|old|archive|trial)[_-]?.*",
]

EPHEMERAL_FILE_REGEXES = [re.compile(pattern) for pattern in EPHEMERAL_FILE_PATTERNS]
EPHEMERAL_DIR_REGEXES = [re.compile(pattern) for pattern in EPHEMERAL_DIR_PATTERNS]

TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 16,
    "bash_timeout": 30,
    "max_output_chars": 8192,
    "workdir": os.environ.get("SLIME_BASH_TOOL_WORKDIR", DEFAULT_WORKDIR),
    "shared_workspace_across_prompts": os.environ.get("SLIME_BASH_SHARED_WORKSPACE_ACROSS_PROMPTS", "true").lower()
    in ("1", "true", "yes", "on"),
    "problem_file": "task.md",
    "trace_dir": os.environ.get("SLIME_BASH_TRACE_DIR", DEFAULT_TRACE_DIR),
    "blocked_patterns": [
        "rm -rf /",
        ":(){ :|:&};:",
        "mkfs",
        "dd if=",
        "> /dev/sd",
    ],
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


class RolloutTracer:
    """Write structured JSONL trace entries for a single rollout."""

    def __init__(self, trace_dir: Path, rollout_key: str | int | None):
        self.rollout_key = rollout_key
        self.trace_file = trace_dir / f"rollout_{rollout_key}_{int(time.time())}.jsonl"
        trace_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, **kwargs):
        entry = {"ts": time.time(), "rollout_key": str(self.rollout_key), "step": step, **kwargs}
        try:
            with open(self.trace_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            logger.warning("Failed to write trace entry: %s", entry)


def create_tracer(rollout_key: str | int | None) -> RolloutTracer | None:
    """Create a tracer using SLIME_BASH_TRACE_DIR (or the default trace directory)."""
    trace_dir = TOOL_CONFIGS.get("trace_dir", "")
    if not trace_dir:
        return None
    return RolloutTracer(Path(trace_dir), rollout_key)


def _truncate(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return output[:half] + f"\n\n... truncated ({len(output)} chars total) ...\n\n" + output[-half:]


class BashSandbox:
    """Execute bash commands with basic safety checks."""

    def __init__(self, timeout: int, max_output_chars: int, blocked_patterns: list[str]):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.blocked_patterns = blocked_patterns

    def _directory_fingerprint(self, workdir: str) -> str:
        """Return an md5 fingerprint representing files under the workdir."""

        root = Path(workdir)
        if not root.exists():
            return ""

        digest = hashlib.md5(usedforsecurity=False)
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue

            rel_path = str(path.relative_to(root))
            digest.update(rel_path.encode("utf-8", errors="replace"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(b":")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\n")

        return digest.hexdigest()

    def _has_file_changes(self, before_fingerprint: str, after_fingerprint: str) -> bool:
        """Return True if any file was created, modified, or deleted."""

        return before_fingerprint != after_fingerprint

    async def execute_command(self, command: str, workdir: str, extra_env: dict[str, str] | None = None) -> str:
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string."

        command = command.strip()
        for pattern in self.blocked_patterns:
            if pattern in command:
                logger.warning("Command blocked by safety policy: %.200s", command)
                return "Error: command blocked by safety policy."

        logger.debug("Executing in %s: %.200s", workdir, command)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        before_fingerprint = self._directory_fingerprint(workdir)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(extra_env or {})},
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning("Command timed out after %ds: %.200s", self.timeout, command)
            return f"Error: command timed out after {self.timeout}s."
        except Exception as e:
            return f"Error: {e}"

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        logger.debug("Exit code=%s, stdout=%d bytes, stderr=%d bytes", proc.returncode, len(stdout_bytes), len(stderr_bytes))

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")

        file_changed = self._has_file_changes(before_fingerprint, self._directory_fingerprint(workdir))
        if file_changed:
            parts.append("Files changed: yes")
            logger.debug("File changes detected in %s", workdir)

        result = "\n".join(parts) if parts else "(no output)"
        return _truncate(result, self.max_output_chars)


class ToolRegistry:
    """Tool registry for a single bash tool."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.base_workdir = Path(TOOL_CONFIGS["workdir"])
        self.main_lock_path = self.base_workdir / "main.lock"
        self._prepare_rollout_workdirs()
        self._rollout_dirs: dict[str, Path] = {}
        self._rollout_base_dirs: dict[str, Path] = {}
        self._rollout_locks: dict[str, asyncio.Lock] = {}
        self.bash_sandbox = BashSandbox(
            timeout=TOOL_CONFIGS["bash_timeout"],
            max_output_chars=TOOL_CONFIGS["max_output_chars"],
            blocked_patterns=TOOL_CONFIGS["blocked_patterns"],
        )
        self._register_default_tools()

    def _prepare_rollout_workdirs(self):
        main_dir = self.base_workdir / "main"
        rollout_root = self.base_workdir / "rollout_envs"
        rollout_base_root = self.base_workdir / "rollout_bases"
        rollout_root.mkdir(parents=True, exist_ok=True)
        rollout_base_root.mkdir(parents=True, exist_ok=True)

        if not main_dir.exists():
            main_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_main_git_repo()

    def get_rollout_lock(self, rollout_key: str | int | None) -> asyncio.Lock:
        key = str(rollout_key) if rollout_key is not None else "_default"
        if key not in self._rollout_locks:
            self._rollout_locks[key] = asyncio.Lock()
        return self._rollout_locks[key]

    def _resolve_rollout_workdir(self, rollout_key: str | int | None) -> Path:
        key = str(rollout_key) if rollout_key is not None else "_default"
        if key not in self._rollout_dirs:
            rollout_dir = self.base_workdir / "rollout_envs" / f"rk_{key}"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            self._rollout_dirs[key] = rollout_dir
        return self._rollout_dirs[key]

    def _resolve_rollout_base_dir(self, rollout_key: str | int | None) -> Path:
        key = str(rollout_key) if rollout_key is not None else "_default"
        if key not in self._rollout_base_dirs:
            rollout_base_dir = self.base_workdir / "rollout_bases" / f"rk_{key}"
            rollout_base_dir.mkdir(parents=True, exist_ok=True)
            self._rollout_base_dirs[key] = rollout_base_dir
        return self._rollout_base_dirs[key]

    def prepare_rollout(self, rollout_key: str | int | None):
        """Refresh a rollout workspace from the latest merged main workspace."""

        main_dir = self.base_workdir / "main"
        rollout_dir = self._resolve_rollout_workdir(rollout_key)
        rollout_base_dir = self._resolve_rollout_base_dir(rollout_key)
        logger.info("[rollout=%s] Preparing workspace: %s", rollout_key, rollout_dir)

        with self._main_workspace_lock():
            self._copy_directory(main_dir, rollout_dir)
            self._copy_directory(main_dir, rollout_base_dir)

    def write_problem_file(self, rollout_key: str | int | None, problem_text: str):
        """Write the per-rollout task description file into the rollout workspace."""

        rollout_dir = self._resolve_rollout_workdir(rollout_key)
        problem_file = rollout_dir / TOOL_CONFIGS["problem_file"]
        problem_file.write_text(problem_text, encoding="utf-8")

    def remove_ephemeral_files(self, rollout_key: str | int | None):
        """Remove per-rollout task and answer files before merge/discard."""

        rollout_dirs = [self._resolve_rollout_workdir(rollout_key)]
        rollout_base_dirs = [self._resolve_rollout_base_dir(rollout_key)]

        main_dir = self.base_workdir / "main"
        managed_roots = [*rollout_dirs, *rollout_base_dirs, main_dir]

        for filename in [TOOL_CONFIGS["problem_file"], REWARD_RESULT_FILE]:
            rel = Path(filename)
            for rollout_dir in rollout_dirs:
                self._write_bytes(rollout_dir, rel, None)
            for rollout_base_dir in rollout_base_dirs:
                self._write_bytes(rollout_base_dir, rel, None)
            self._write_bytes(main_dir, rel, None)

        for root in managed_roots:
            self._remove_pattern_matched_ephemeral_artifacts(root)

    def _matches_ephemeral_file_pattern(self, name: str) -> bool:
        return any(regex.fullmatch(name) for regex in EPHEMERAL_FILE_REGEXES)

    def _matches_ephemeral_dir_pattern(self, name: str) -> bool:
        return any(regex.fullmatch(name) for regex in EPHEMERAL_DIR_REGEXES)

    def _is_empty_file(self, path: Path) -> bool:
        try:
            return path.stat().st_size == 0
        except OSError:
            return False

    def _has_no_extension(self, path: Path) -> bool:
        return path.suffix == ""

    def _remove_pattern_matched_ephemeral_artifacts(self, root: Path):
        if not root.exists():
            return

        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if ".git" in path.parts:
                continue
            if path.is_file() and (
                self._matches_ephemeral_file_pattern(path.name)
                or self._is_empty_file(path)
                or self._has_no_extension(path)
            ):
                path.unlink(missing_ok=True)
            elif path.is_dir() and self._matches_ephemeral_dir_pattern(path.name):
                shutil.rmtree(path, ignore_errors=True)

    @contextmanager
    def _main_workspace_lock(self):
        self.main_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.main_lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _copy_directory(self, source: Path, target: Path):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)

    def _iter_files(self, root: Path) -> set[Path]:
        files: set[Path] = set()
        if not root.exists():
            return files
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                files.add(path.relative_to(root))
        return files

    def _read_bytes(self, root: Path, relative_path: Path) -> bytes | None:
        path = root / relative_path
        if not path.exists() or not path.is_file():
            return None
        return path.read_bytes()

    def _write_bytes(self, root: Path, relative_path: Path, content: bytes | None):
        path = root / relative_path
        if content is None:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _merge_text_with_git(self, base: bytes, current: bytes, incoming: bytes) -> tuple[bytes, bool]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            current_file = tmp / "current.txt"
            base_file = tmp / "base.txt"
            incoming_file = tmp / "incoming.txt"
            current_file.write_bytes(current)
            base_file.write_bytes(base)
            incoming_file.write_bytes(incoming)

            proc = subprocess.run(
                [
                    "git",
                    "merge-file",
                    "-p",
                    "--diff3",
                    "-L",
                    "main",
                    "-L",
                    "base",
                    "-L",
                    "rollout",
                    str(current_file),
                    str(base_file),
                    str(incoming_file),
                ],
                capture_output=True,
                check=False,
            )
        if proc.returncode in (0, 1) and proc.stdout:
            return proc.stdout, proc.returncode == 1
        return incoming, True

    def _build_conflict_copy(self, source_name: str, content: bytes | None) -> bytes:
        if content is None:
            return f"<{source_name} deleted this file>\n".encode("utf-8")
        return content

    def _ensure_main_git_repo(self):
        main_dir = self.base_workdir / "main"
        if not (main_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=main_dir, check=True)
            subprocess.run(["git", "config", "user.email", "slime-bash@example.com"], cwd=main_dir, check=True)
            subprocess.run(["git", "config", "user.name", "slime-bash-tool"], cwd=main_dir, check=True)
            subprocess.run(["git", "add", "-A"], cwd=main_dir, check=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "Initialize main workspace"], cwd=main_dir, check=True)

    def finalize_rollout(self, rollout_key: str | int | None, reward: float | int) -> str:
        key = str(rollout_key) if rollout_key is not None else "_default"
        rollout_dir = self._resolve_rollout_workdir(rollout_key)
        rollout_base_dir = self._resolve_rollout_base_dir(rollout_key)
        main_dir = self.base_workdir / "main"

        try:
            with self._main_workspace_lock():
                self.remove_ephemeral_files(rollout_key)
                reward_value = float(reward)
                logger.info("[rollout=%s] Finalizing: reward=%.4f", rollout_key, reward_value)
                if reward_value <= 0:
                    msg = f"Discarded rollout changes for reward={reward_value:.4f}."
                    logger.info("[rollout=%s] %s", rollout_key, msg)
                    return msg

                all_files = self._iter_files(main_dir) | self._iter_files(rollout_dir) | self._iter_files(rollout_base_dir)
                conflict_count = 0
                for rel in sorted(all_files):
                    base_content = self._read_bytes(rollout_base_dir, rel)
                    main_content = self._read_bytes(main_dir, rel)
                    rollout_content = self._read_bytes(rollout_dir, rel)

                    if rollout_content == base_content:
                        continue

                    if main_content == base_content:
                        self._write_bytes(main_dir, rel, rollout_content)
                        continue

                    if main_content == rollout_content:
                        continue

                    if base_content is not None and main_content is not None and rollout_content is not None:
                        try:
                            merged, has_conflict = self._merge_text_with_git(base_content, main_content, rollout_content)
                            self._write_bytes(main_dir, rel, merged)
                            if has_conflict:
                                conflict_count += 1
                                self._write_bytes(
                                    main_dir,
                                    rel.with_suffix(rel.suffix + ".main"),
                                    self._build_conflict_copy("main", main_content),
                                )
                                self._write_bytes(
                                    main_dir,
                                    rel.with_suffix(rel.suffix + ".rollout"),
                                    self._build_conflict_copy("rollout", rollout_content),
                                )
                            continue
                        except UnicodeDecodeError:
                            continue

                    conflict_count += 1
                    self._write_bytes(main_dir, rel.with_suffix(rel.suffix + ".main"), self._build_conflict_copy("main", main_content))
                    self._write_bytes(main_dir, rel.with_suffix(rel.suffix + ".rollout"), self._build_conflict_copy("rollout", rollout_content))

                logger.info("[rollout=%s] Merge complete: %d files examined, %d conflicts", rollout_key, len(all_files), conflict_count)
                subprocess.run(["git", "add", "-A"], cwd=main_dir, check=True)
                commit_msg = f"Merge rollout {rollout_key} (reward={reward_value:.4f}, conflicts={conflict_count})"
                commit = subprocess.run(["git", "commit", "-m", commit_msg], cwd=main_dir, capture_output=True, text=True, check=False)
                logger.debug("[rollout=%s] Git commit returncode=%d, stdout=%.200s", rollout_key, commit.returncode, commit.stdout or "")
                if commit.returncode != 0 and "nothing to commit" in commit.stdout + commit.stderr:
                    msg = f"No merge changes from rollout reward={reward_value:.4f}."
                    logger.info("[rollout=%s] %s", rollout_key, msg)
                    return msg
                if commit.returncode != 0:
                    msg = f"Merge failed to commit: {commit.stderr.strip()}"
                    logger.warning("[rollout=%s] %s", rollout_key, msg)
                    return msg
                msg = f"Merged rollout with reward={reward_value:.4f}; conflicts={conflict_count}."
                logger.info("[rollout=%s] %s", rollout_key, msg)
                return msg
        finally:
            for rollout_path in [rollout_dir, rollout_base_dir]:
                if rollout_path.exists():
                    shutil.rmtree(rollout_path)
            self._rollout_dirs.pop(key, None)
            self._rollout_base_dirs.pop(key, None)
            self._rollout_locks.pop(key, None)

    def _register_default_tools(self):
        self.register_tool(
            "bash",
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": (
                        "Run a command in a bash shell. "
                        "The shell can access CLI tools like git, python, cat, and pytest."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], rollout_key: str | int | None = None) -> str:
        if tool_name != "bash":
            return f"Error: Tool '{tool_name}' not found"

        command = arguments.get("command", "")
        workdir = self._resolve_rollout_workdir(rollout_key)
        extra_env = {
            "SLIME_BASH_MAIN_DIR": str(workdir),
            "SLIME_BASH_ROLLOUT_KEY": "" if rollout_key is None else str(rollout_key),
        }
        async with SEMAPHORE:
            return await self.bash_sandbox.execute_command(command, workdir=str(workdir), extra_env=extra_env)


tool_registry = ToolRegistry()
