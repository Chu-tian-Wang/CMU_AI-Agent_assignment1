"""Run sandboxes as local Docker containers instead of Modal sandboxes.

Selected with `SANDBOX_BACKEND=docker`. Each `Environment` is one container
whose main process is `sleep`, so it is reclaimed after `deployment_timeout`
seconds just as a Modal sandbox is. Commands run through `docker exec` with the
semantics of SWE-ReX's `Command`, so nothing extra is installed in the image.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from swerex.exceptions import CommandTimeoutError, NonZeroExitCodeError
from swerex.runtime.abstract import Command, CommandResponse

DOCKER = "docker"
LABEL = "assignment.sandbox"
REPOSITORY = "assignment-sandbox"
# Seconds between the in-container SIGTERM and SIGKILL when a command times out.
KILL_AFTER = 5
# Extra seconds the docker CLI gets, so the in-container `timeout` fires first.
CLIENT_GRACE = 15
# Exit codes of `timeout` when it had to stop the command: TERM, then KILL.
TIMED_OUT = (124, 137)


def _docker(*args: str, timeout: float | None = 60, capture: bool = True) -> subprocess.CompletedProcess:
    """Run one docker CLI command. Uncaptured output streams to the console."""
    return subprocess.run(
        [DOCKER, *args],
        capture_output=capture,
        text=True,
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )


def docker_status() -> tuple[bool, str]:
    """Whether the daemon answers, with its version and arch or the error."""
    try:
        result = _docker("version", "--format", "{{.Server.Version}} {{.Server.Arch}}", timeout=20)
    except FileNotFoundError:
        return False, "the docker CLI is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return False, "the docker daemon did not answer within 20s"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
    return True, result.stdout.strip()


def image_exists(image: str) -> bool:
    return _docker("image", "inspect", "--format", "{{.Id}}", image).returncode == 0


def image_id(image: str) -> str:
    result = _docker("image", "inspect", "--format", "{{.Id}}", image)
    if result.returncode != 0:
        raise RuntimeError(f"Image {image} is not available locally: {result.stderr.strip()}")
    return result.stdout.strip()


def ensure_pulled(image: str) -> str:
    """Pull an image unless it is already local. Large images take a while."""
    if not image_exists(image):
        if _docker("pull", image, timeout=None, capture=False).returncode != 0:
            raise RuntimeError(f"docker pull {image} failed")
    return image


def build_image(
    dockerfile: str,
    files: Iterable[tuple[Path, str]],
    name: str,
    force_build: bool = False,
) -> str:
    """Build an image from Dockerfile text and (local file, context path) pairs.

    The tag is a digest of every input, so an unchanged build is reused instead
    of rebuilt, and a changed source can never be mistaken for a cached one.

    Returns:
        The image tag.
    """
    files = sorted(files, key=lambda item: item[1])
    digest = hashlib.sha256(dockerfile.encode())
    for source, target in files:
        digest.update(f"\0{target}\0".encode())
        digest.update(Path(source).read_bytes())
    tag = f"{REPOSITORY}:{name}-{digest.hexdigest()[:16]}"
    if not force_build and image_exists(tag):
        return tag

    with tempfile.TemporaryDirectory(prefix="assignment-build-") as tmp:
        context = Path(tmp) / "context"
        context.mkdir()
        for source, target in files:
            destination = context / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        # Outside the context, so `COPY . <dir>` cannot pick it up.
        dockerfile_path = Path(tmp) / "Dockerfile"
        dockerfile_path.write_text(dockerfile, newline="\n")
        args = ["build", "-t", tag, "-f", str(dockerfile_path)]
        if force_build:
            args.append("--no-cache")
        if _docker(*args, str(context), timeout=None, capture=False).returncode != 0:
            raise RuntimeError(f"docker build of {tag} failed; see the build output above")
    return tag


@dataclass(frozen=True)
class DockerImage:
    """An image plus local files layered on top: the part of `modal.Image` the
    sandboxes use. Nothing is built until `build` is called."""

    base: str
    files: tuple[tuple[str, str], ...] = ()

    def add_local_file(self, local_path: str, remote_path: str, copy: bool = True) -> DockerImage:
        """Copy a local file into the image, as `modal.Image.add_local_file` does."""
        return DockerImage(self.base, (*self.files, (str(local_path), remote_path)))

    def build(self) -> str:
        """Pull or build everything this image needs and return its tag."""
        ensure_pulled(self.base)
        if not self.files:
            return self.base
        # The base id goes into the Dockerfile, so a rebuilt base gets a new tag.
        lines = [f"# base {image_id(self.base)}", f"FROM {self.base}"]
        context = []
        for index, (local_path, remote_path) in enumerate(self.files):
            name = f"{index}-{Path(local_path).name}"
            lines.append(f"COPY {name} {remote_path}")
            context.append((Path(local_path), name))
        return build_image("\n".join(lines) + "\n", context, name="files")


class DockerContainer:
    """One running container, standing in for a Modal sandbox."""

    def __init__(
        self,
        image: str | DockerImage,
        deployment_timeout: float = 600,
        runtime_timeout: float = 600,
        startup_timeout: float = 600,
        ports: Iterable[int] = (),
    ):
        if not isinstance(image, (str, DockerImage)):
            raise TypeError(
                "The docker backend takes an image name or a DockerImage from "
                f"build_testbed_image, not {type(image).__name__}."
            )
        self.image = image
        self.deployment_timeout = deployment_timeout
        self.runtime_timeout = runtime_timeout
        self.startup_timeout = startup_timeout
        self.ports = list(dict.fromkeys(ports))
        self.name: str | None = None

    def start(self) -> None:
        tag = self.image.build() if isinstance(self.image, DockerImage) else ensure_pulled(self.image)
        name = f"assignment-{uuid.uuid4().hex[:12]}"
        args = ["run", "--detach", "--rm", "--init", "--name", name, "--label", f"{LABEL}=1"]
        for port in self.ports:
            # Loopback only: the chess API has no authentication.
            args += ["--publish", f"127.0.0.1::{port}"]
        # `sleep` as the main process reclaims the container after
        # deployment_timeout, like Modal, even if this process dies first.
        args += ["--entrypoint", "sleep", tag, str(max(1, int(self.deployment_timeout)))]
        try:
            result = _docker(*args, timeout=self.startup_timeout)
        except subprocess.TimeoutExpired as exc:
            _docker("rm", "-f", name)
            raise RuntimeError(f"Container did not start within {self.startup_timeout:g}s") from exc
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")
        self.name = name

    def running(self) -> bool:
        if self.name is None:
            return False
        result = _docker("container", "inspect", "--format", "{{.State.Running}}", self.name, timeout=30)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def port_url(self, port: int) -> str:
        """The host URL a published container port is reachable at."""
        if port not in self.ports:
            forwarded = ", ".join(str(item) for item in sorted(self.ports)) or "none"
            raise ValueError(f"Port {port} was not forwarded. Available forwarded ports: {forwarded}.")
        result = _docker("port", self._require_name(), f"{port}/tcp")
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"Could not read the host port for {port}: {result.stderr.strip()}")
        host_port = result.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        return f"http://127.0.0.1:{host_port}"

    def execute(self, command: Command) -> CommandResponse:
        """Run one command with `docker exec`, following SWE-ReX's semantics.

        Raises:
            CommandTimeoutError: The command ran past its timeout.
            NonZeroExitCodeError: It failed and `command.check` was set.
            RuntimeError: The container is gone.
        """
        name = self._require_name()
        limit = self.runtime_timeout if command.timeout is None else command.timeout
        limit = max(float(limit), 0.001)

        args = [DOCKER, "exec"]
        if command.cwd:
            args += ["--workdir", command.cwd]
        for key, value in (command.env or {}).items():
            args += ["--env", f"{key}={value}"]
        args.append(name)
        # Kill the command inside the container too; stopping the docker CLI
        # alone would leave it running there.
        args += ["/usr/bin/timeout", "--kill-after", str(KILL_AFTER), f"{limit:g}"]
        if command.shell:
            # Same argv as subprocess.run(..., shell=True).
            argv = [command.command] if isinstance(command.command, str) else list(command.command)
            args += ["/bin/sh", "-c", *argv]
        else:
            args += [command.command] if isinstance(command.command, str) else list(command.command)

        started = time.monotonic()
        try:
            result = subprocess.run(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if command.merge_output_streams else subprocess.PIPE,
                timeout=limit + KILL_AFTER + CLIENT_GRACE,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandTimeoutError(f"Timeout ({limit:g}s) exceeded while running command") from exc
        if result.returncode in TIMED_OUT and time.monotonic() - started >= limit:
            raise CommandTimeoutError(f"Timeout ({limit:g}s) exceeded while running command")

        response = CommandResponse(
            stdout=result.stdout.decode(errors="backslashreplace"),
            stderr=result.stderr.decode(errors="backslashreplace") if result.stderr is not None else "",
            exit_code=result.returncode,
        )
        # A dead container fails every exec with a daemon error, and the exit
        # code alone does not tell that apart from a failing command.
        if result.returncode != 0 and not self.running():
            raise RuntimeError(f"Container {name} is no longer running: {response.stderr.strip()}")
        if command.check and result.returncode != 0:
            message = (
                f"Command {command.command!r} failed with exit code {result.returncode}. "
                f"Stdout:\n{response.stdout!r}\nStderr:\n{response.stderr!r}"
            )
            if command.error_msg:
                message = f"{command.error_msg}: {message}"
            raise NonZeroExitCodeError(message)
        return response

    def stop(self, timeout: float = 30) -> None:
        """Remove the container. Safe to call more than once."""
        if self.name is None:
            return
        result = _docker("rm", "-f", self.name, timeout=timeout)
        gone = "No such container" in result.stderr or "already in progress" in result.stderr
        if result.returncode != 0 and not gone:
            raise RuntimeError(f"Could not remove container {self.name}: {result.stderr.strip()}")
        self.name = None

    def _require_name(self) -> str:
        if self.name is None:
            raise RuntimeError("The container is not running.")
        return self.name
