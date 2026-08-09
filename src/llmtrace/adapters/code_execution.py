"""Code execution backend for coding benchmark tasks (HumanEval).

Provides a secure sandbox abstraction for executing model-generated code.
The default implementation uses Docker with strict resource limits.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class CodeExecutionError(Exception):
    """Base exception for code execution failures."""


class SandboxUnavailableError(CodeExecutionError):
    """The sandbox backend is not available (infrastructure failure)."""


class CodeExecutionResult:
    """Result of a single code execution in the sandbox."""

    __slots__ = ("success", "stdout", "stderr", "exit_code", "timed_out")

    def __init__(
        self,
        *,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = -1,
        timed_out: bool = False,
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out


class CodeExecutionBackend(ABC):
    """Abstract sandbox for executing model-generated code.

    Implementations MUST be secure — model code must never access the
    host filesystem, network, or credentials.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the sandbox backend is available."""
        ...

    @abstractmethod
    def execute(
        self,
        code: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> CodeExecutionResult:
        """Execute *code* in the sandbox and return the result.

        Args:
            code: The complete Python code to execute.
            timeout_seconds: Wall-clock timeout for execution.

        Returns:
            CodeExecutionResult with stdout/stderr and exit status.

        Raises:
            SandboxUnavailableError: If the sandbox is not available.
        """
        ...


# ---------------------------------------------------------------------------
# In-process (UNSAFE — only for tests with trusted code)
# ---------------------------------------------------------------------------


class _InProcessExecutionBackend(CodeExecutionBackend):
    """UNSAFE in-process executor for unit tests only.

    This backend runs code directly in the current Python process.
    It MUST NOT be used with untrusted model-generated code.
    """

    def is_available(self) -> bool:
        return True

    def execute(
        self,
        code: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> CodeExecutionResult:
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            out_buf = StringIO()
            err_buf = StringIO()
            sys.stdout = out_buf
            sys.stderr = err_buf
            compiled = compile(code, "<sandbox>", "exec")
            ns: dict[str, Any] = {}
            exec(compiled, ns)  # noqa: S102 — only for trusted test code
            return CodeExecutionResult(
                success=True,
                stdout=out_buf.getvalue(),
                stderr=err_buf.getvalue(),
                exit_code=0,
                timed_out=False,
            )
        except Exception as exc:
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=1,
                timed_out=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Docker sandbox
# ---------------------------------------------------------------------------


class DockerCodeExecutionBackend(CodeExecutionBackend):
    """Docker-based secure code execution sandbox.

    Each execution runs in a fresh container with:
      - --network none
      - --read-only root filesystem
      - Non-root user (nobody)
      - Limited CPU and memory
      - No host volume mounts
      - Stripped environment variables

    If Docker is not available, is_available() returns False
    and execute() raises SandboxUnavailableError.
    """

    def __init__(
        self,
        *,
        cpu_limit: str = "1.0",
        memory_limit: str = "256m",
        pids_limit: int = 50,
        image: str = "python:3.12-slim",
    ) -> None:
        self._cpu_limit = cpu_limit
        self._memory_limit = memory_limit
        self._pids_limit = pids_limit
        self._image = image
        self._checked_available: bool | None = None

    def is_available(self) -> bool:
        if self._checked_available is not None:
            return self._checked_available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            self._checked_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._checked_available = False
        return self._checked_available

    def execute(
        self,
        code: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> CodeExecutionResult:
        if not self.is_available():
            raise SandboxUnavailableError("Docker is not available for code execution")

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        with tempfile.TemporaryDirectory(prefix=f"llmtrace-sandbox-{code_hash}-") as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "user_code.py"
            test_path = tmp_path / "run_tests.py"
            script_path.write_text(code)

            # Wrapper that runs the user code file
            test_path.write_text(code)

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--tmpfs",
                "/tmp:exec",
                "--user=nobody",
                f"--cpus={self._cpu_limit}",
                f"--memory={self._memory_limit}",
                f"--pids-limit={str(self._pids_limit)}",
                "-v",
                f"{tmp_path}:/code:ro",
                "-w",
                "/code",
                "--env-file",
                "/dev/null",
                self._image,
                "python3",
                "/code/user_code.py",
            ]

            try:
                proc = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    timeout=timeout_seconds + 5,
                    check=False,
                )
                timed_out = False
            except subprocess.TimeoutExpired:
                return CodeExecutionResult(
                    success=False,
                    stdout="",
                    stderr="execution timed out",
                    exit_code=-1,
                    timed_out=True,
                )

            return CodeExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout.decode("utf-8", errors="replace")[:4096],
                stderr=proc.stderr.decode("utf-8", errors="replace")[:4096],
                exit_code=proc.returncode,
                timed_out=timed_out,
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_code_execution_backend(docker: bool = True) -> CodeExecutionBackend:
    """Return the best available code execution backend.

    If *docker* is True and Docker is available, returns a
    DockerCodeExecutionBackend.  Otherwise returns _InProcessExecutionBackend
    (UNSAFE — for tests only).
    """
    if docker:
        docker_backend = DockerCodeExecutionBackend()
        if docker_backend.is_available():
            return docker_backend
    return _InProcessExecutionBackend()
