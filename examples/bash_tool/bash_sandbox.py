"""
Bash sandbox module for safe shell command execution in multi-turn RL.

Provides isolated bash execution environments for parallel rollouts,
with per-slot working directories, command safety checks, timeout
enforcement, and output truncation.

Each rollout gets its own slot directory so concurrent rollouts
don't interfere with each other.
"""

import asyncio
import os
import shutil
import subprocess
import threading
from typing import Any

# Configuration for bash tool execution
BASH_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,
    # Bash execution settings
    "timeout": 30,
    "max_output_chars": 8192,
    "num_slots": 8,
    "workdir": "/tmp/slime_bash",
    # Safety settings
    "blocked_patterns": [
        "rm -rf /",
        ":(){ :|:&};:",
        "mkfs",
        "dd if=",
        "> /dev/sd",
        "chmod -R 777 /",
        "curl | bash",
        "wget | bash",
    ],
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(BASH_CONFIGS["tool_concurrency"])


def _truncate(output: str, max_chars: int) -> str:
    """Truncate output symmetrically, keeping head and tail."""
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        output[:half]
        + f"\n\n... truncated ({len(output)} chars total) ...\n\n"
        + output[-half:]
    )


class BashSlotPool:
    """
    Manages a pool of isolated working directories (slots) for parallel rollouts.

    Each slot is a separate directory. Before a batch of rollouts begins,
    call `snapshot_to_slots(source_dir)` to copy the task repo into all slots.
    During rollouts, each concurrent instance claims a slot via `acquire()`
    and releases it via `release()`.
    """

    def __init__(self, base_workdir: str, num_slots: int):
        self.base_workdir = base_workdir
        self.num_slots = num_slots

        self._lock = threading.Lock()
        self._free_slots: list[int] = list(range(num_slots))
        self._slot_dirs: dict[int, str] = {}

        for i in range(num_slots):
            d = os.path.join(base_workdir, f"slot_{i}")
            os.makedirs(d, exist_ok=True)
            self._slot_dirs[i] = d

    def get_slot_dir(self, slot: int) -> str:
        return self._slot_dirs[slot]

    def acquire(self) -> tuple[int, str] | None:
        """Acquire a free slot. Returns (slot_id, slot_dir) or None if all busy."""
        with self._lock:
            if not self._free_slots:
                return None
            slot = self._free_slots.pop(0)
            return slot, self._slot_dirs[slot]

    def release(self, slot: int) -> None:
        """Release a slot back to the pool."""
        with self._lock:
            if slot not in self._free_slots:
                self._free_slots.append(slot)

    def snapshot_to_slots(self, source_dir: str) -> None:
        """
        Copy a source repo into all slot directories.
        Call BEFORE starting parallel rollouts for a new task/prompt.
        """
        for i, slot_dir in self._slot_dirs.items():
            if os.path.realpath(slot_dir) == os.path.realpath(source_dir):
                continue
            # Use git clone --local if it's a git repo, else rsync
            if os.path.isdir(os.path.join(source_dir, ".git")):
                if os.path.exists(slot_dir):
                    shutil.rmtree(slot_dir)
                subprocess.run(
                    ["git", "clone", "--local", "--no-hardlinks", source_dir, slot_dir],
                    check=True,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    ["rsync", "-a", "--delete", f"{source_dir}/", f"{slot_dir}/"],
                    check=True,
                    capture_output=True,
                )


# Global slot pool
_slot_pool = BashSlotPool(
    base_workdir=BASH_CONFIGS["workdir"],
    num_slots=BASH_CONFIGS["num_slots"],
)


class BashExecutor:
    """
    Executes bash commands in an isolated slot directory.

    Usage:
        executor = BashExecutor()
        executor.acquire_slot()
        result = await executor.execute("ls -la")
        executor.release_slot()
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_output_chars: int | None = None,
        blocked_patterns: list[str] | None = None,
    ):
        self.timeout = timeout or BASH_CONFIGS["timeout"]
        self.max_output_chars = max_output_chars or BASH_CONFIGS["max_output_chars"]
        self.blocked_patterns = blocked_patterns or BASH_CONFIGS["blocked_patterns"]

        self._slot: int | None = None
        self._workdir: str | None = None

    @property
    def workdir(self) -> str:
        if self._workdir is None:
            # Fallback to slot_0 if no slot acquired
            return _slot_pool.get_slot_dir(0)
        return self._workdir

    def acquire_slot(self) -> bool:
        """Acquire a working directory slot. Returns True if successful."""
        result = _slot_pool.acquire()
        if result is None:
            return False
        self._slot, self._workdir = result
        return True

    def release_slot(self) -> None:
        """Release the working directory slot back to the pool."""
        if self._slot is not None:
            _slot_pool.release(self._slot)
            self._slot = None
            self._workdir = None

    def _check_command_safety(self, command: str) -> str | None:
        """Check if command contains blocked patterns. Returns error message or None."""
        for pattern in self.blocked_patterns:
            if pattern in command:
                return f"Error: command blocked by safety policy (matched: '{pattern}')."
        return None

    async def execute(self, command: str) -> tuple[str, bool]:
        """
        Execute a bash command.

        Returns:
            (output_text, success) tuple
        """
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string.", False

        command = command.strip()

        # Safety check
        safety_error = self._check_command_safety(command)
        if safety_error is not None:
            return safety_error, False

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workdir,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                ),
                timeout=2,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
            exit_code = proc.returncode
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

        except asyncio.TimeoutError:
            return f"Error: command timed out after {self.timeout}s.", False
        except Exception as e:
            return f"Error: {e}", False

        # Format output
        parts = []
        if stdout.strip():
            parts.append(stdout.strip())
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr.strip()}")
        if exit_code != 0:
            parts.append(f"Exit code: {exit_code}")

        output = "\n".join(parts) if parts else "(no output)"
        output = _truncate(output, self.max_output_chars)

        return output, (exit_code == 0)


class BashToolRegistry:
    """
    Tool registry exposing bash as an OpenAI-compatible function tool.

    This provides the tool schema for the LLM and dispatches tool calls
    to BashExecutor instances.
    """

    TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a command in a bash shell. "
                "The shell has access to the working repository. "
                "Use standard CLI tools: git, python, pytest, cat, grep, sed, awk, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    }

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get tool specifications list for LLM."""
        return [self.TOOL_SCHEMA]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], executor: BashExecutor) -> str:
        """Execute a tool call using the provided executor."""
        if tool_name != "bash":
            return f"Error: unknown tool '{tool_name}'."
        command = arguments.get("command", "")
        output, _success = await executor.execute(command)
        return output


# Global registry instance
tool_registry = BashToolRegistry()

# Expose slot pool for snapshot operations
slot_pool = _slot_pool
