"""Bash tool sandbox for multi-turn RLVR environments."""

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 16,
    "bash_timeout": 30,
    "max_output_chars": 8192,
    "workdir": "/tmp/slime_bash_tool",
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

    def __init__(self, timeout: int, max_output_chars: int, workdir: str, blocked_patterns: list[str]):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.workdir = workdir
        self.blocked_patterns = blocked_patterns
        Path(self.workdir).mkdir(parents=True, exist_ok=True)

    def _directory_fingerprint(self) -> str:
        """Return an md5 fingerprint representing files under the workdir."""

        root = Path(self.workdir)
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

    async def execute_command(self, command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string."

        command = command.strip()
        for pattern in self.blocked_patterns:
            if pattern in command:
                return "Error: command blocked by safety policy."

        before_fingerprint = self._directory_fingerprint()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
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

        if self._has_file_changes(before_fingerprint, self._directory_fingerprint()):
            parts.append("Files changed: yes")

        result = "\n".join(parts) if parts else "(no output)"
        return _truncate(result, self.max_output_chars)


class ToolRegistry:
    """Tool registry for a single bash tool."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.bash_sandbox = BashSandbox(
            timeout=TOOL_CONFIGS["bash_timeout"],
            max_output_chars=TOOL_CONFIGS["max_output_chars"],
            workdir=TOOL_CONFIGS["workdir"],
            blocked_patterns=TOOL_CONFIGS["blocked_patterns"],
        )
        self._register_default_tools()

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

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name != "bash":
            return f"Error: Tool '{tool_name}' not found"
        command = arguments.get("command", "")
        return await self.bash_sandbox.execute_command(command)


tool_registry = ToolRegistry()
