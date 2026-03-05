"""Bash tool sandbox for multi-turn RLVR environments."""

import asyncio
import hashlib
import os
import subprocess
import shutil
from pathlib import Path
from typing import Any

DEFAULT_WORKDIR = "/opt/NeMo/slime_bash_tool_workspace"

TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 16,
    "bash_timeout": 30,
    "max_output_chars": 8192,
    "workdir": os.environ.get("SLIME_BASH_TOOL_WORKDIR", DEFAULT_WORKDIR),
    "num_rollout_envs": 8,
    "shared_workspace_across_prompts": True,
    "blocked_patterns": [
        "rm -rf /",
        ":(){ :|:&};:",
        "mkfs",
        "dd if=",
        "> /dev/sd",
    ],
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


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
                return "Error: command blocked by safety policy."

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
            return f"Error: command timed out after {self.timeout}s."
        except Exception as e:
            return f"Error: {e}"

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")

        if self._has_file_changes(before_fingerprint, self._directory_fingerprint(workdir)):
            parts.append("Files changed: yes")

        result = "\n".join(parts) if parts else "(no output)"
        return _truncate(result, self.max_output_chars)


class ToolRegistry:
    """Tool registry for a single bash tool."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.base_workdir = Path(TOOL_CONFIGS["workdir"])
        self.num_rollout_envs = int(TOOL_CONFIGS["num_rollout_envs"])
        self.rollout_workdirs = self._prepare_rollout_workdirs()
        self.rollout_locks = [asyncio.Lock() for _ in range(self.num_rollout_envs)]
        self.bash_sandbox = BashSandbox(
            timeout=TOOL_CONFIGS["bash_timeout"],
            max_output_chars=TOOL_CONFIGS["max_output_chars"],
            blocked_patterns=TOOL_CONFIGS["blocked_patterns"],
        )
        self._register_default_tools()

    def _prepare_rollout_workdirs(self) -> list[Path]:
        main_dir = self.base_workdir / "main"
        rollout_root = self.base_workdir / "rollout_envs"
        rollout_base_root = self.base_workdir / "rollout_bases"
        rollout_root.mkdir(parents=True, exist_ok=True)
        rollout_base_root.mkdir(parents=True, exist_ok=True)

        if not main_dir.exists():
            main_dir.mkdir(parents=True, exist_ok=True)

        rollout_workdirs: list[Path] = []
        for idx in range(self.num_rollout_envs):
            target_dir = rollout_root / f"main_{idx}"
            self._reset_rollout_dir(target_dir)
            rollout_workdirs.append(target_dir)

            base_dir = rollout_base_root / f"main_{idx}"
            self._copy_directory(main_dir, base_dir)

        self._ensure_main_git_repo()

        return rollout_workdirs

    def _resolve_rollout_slot(self, rollout_key: str | int | None) -> int:
        """Select the rollout slot index used for workspace isolation."""

        if TOOL_CONFIGS.get("shared_workspace_across_prompts", True):
            return 0
        if rollout_key is None:
            return 0
        return int(rollout_key) % len(self.rollout_workdirs)

    def get_rollout_lock(self, rollout_key: str | int | None) -> asyncio.Lock:
        return self.rollout_locks[self._resolve_rollout_slot(rollout_key)]

    def _resolve_rollout_workdir(self, rollout_key: str | int | None) -> Path:
        return self.rollout_workdirs[self._resolve_rollout_slot(rollout_key)]

    def _resolve_rollout_base_dir(self, rollout_key: str | int | None) -> Path:
        idx = self._resolve_rollout_slot(rollout_key)
        return self.base_workdir / "rollout_bases" / f"main_{idx}"

    def prepare_rollout(self, rollout_key: str | int | None):
        """Refresh a rollout workspace from the latest merged main workspace."""

        main_dir = self.base_workdir / "main"
        rollout_dir = self._resolve_rollout_workdir(rollout_key)
        rollout_base_dir = self._resolve_rollout_base_dir(rollout_key)
        self._copy_directory(main_dir, rollout_dir)
        self._copy_directory(main_dir, rollout_base_dir)

    def _copy_directory(self, source: Path, target: Path):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)

    def _reset_rollout_dir(self, target_dir: Path):
        self._copy_directory(self.base_workdir / "main", target_dir)

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
        rollout_dir = self._resolve_rollout_workdir(rollout_key)
        rollout_base_dir = self._resolve_rollout_base_dir(rollout_key)
        main_dir = self.base_workdir / "main"

        reward_value = float(reward)
        if reward_value <= 0:
            self._reset_rollout_dir(rollout_dir)
            self._copy_directory(main_dir, rollout_base_dir)
            return f"Discarded rollout changes for reward={reward_value:.4f}."

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
                    pass

            conflict_count += 1
            self._write_bytes(main_dir, rel.with_suffix(rel.suffix + ".main"), self._build_conflict_copy("main", main_content))
            self._write_bytes(main_dir, rel.with_suffix(rel.suffix + ".rollout"), self._build_conflict_copy("rollout", rollout_content))

        subprocess.run(["git", "add", "-A"], cwd=main_dir, check=True)
        commit_msg = f"Merge rollout {rollout_key} (reward={reward_value:.4f}, conflicts={conflict_count})"
        commit = subprocess.run(["git", "commit", "-m", commit_msg], cwd=main_dir, capture_output=True, text=True, check=False)
        self._reset_rollout_dir(rollout_dir)
        self._copy_directory(main_dir, rollout_base_dir)
        if commit.returncode != 0 and "nothing to commit" in commit.stdout + commit.stderr:
            return f"No merge changes from rollout reward={reward_value:.4f}."
        if commit.returncode != 0:
            return f"Merge failed to commit: {commit.stderr.strip()}"
        return f"Merged rollout with reward={reward_value:.4f}; conflicts={conflict_count}."

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
        return await self.bash_sandbox.execute_command(command, workdir=str(workdir), extra_env=extra_env)


tool_registry = ToolRegistry()
