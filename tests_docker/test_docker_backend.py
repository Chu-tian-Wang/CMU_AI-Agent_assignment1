"""Unit tests for the docker backend. The docker CLI is faked, so these run anywhere."""

import subprocess
import time
from pathlib import Path

import pytest

from assignment import docker_backend
from assignment.docker_backend import DockerImage
from assignment.env import Environment, sandbox_backend
from assignment.task import Task
from assignment.utils import image as image_module

ROOT = Path(__file__).resolve().parents[1]
# git still has to run for real, for verify_source.
REAL_RUN = subprocess.run


class FakeDocker:
    """Stands in for `subprocess.run` inside docker_backend.

    Handlers are keyed by docker subcommand and return (exit code, stdout,
    stderr) as text; exec output is turned into bytes as the real CLI gives.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self.handlers = {
            "image": lambda args: (0, "sha256:base\n", ""),
            "pull": lambda args: (0, "", ""),
            "build": lambda args: (0, "", ""),
            "run": lambda args: (0, "abc123\n", ""),
            "container": lambda args: (0, "true\n", ""),
            "port": lambda args: (0, "127.0.0.1:49153\n", ""),
            "rm": lambda args: (0, "", ""),
            "exec": self.exec,
        }

    def exec(self, args):
        if "uname -s; uname -r; uname -v; uname -m" in args:
            return 0, "Linux\n6.8.0\n#1 SMP\nx86_64\n", ""
        return 0, "", ""

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        if args[0] != docker_backend.DOCKER:
            return REAL_RUN(args, **kwargs)
        code, out, err = self.handlers[args[1]](args)
        if not kwargs.get("text"):
            out, err = out.encode(), err.encode()
        return subprocess.CompletedProcess(args, code, out, err)

    def last(self, subcommand: str) -> list[str]:
        return [call for call in self.calls if call[1] == subcommand][-1]


@pytest.fixture
def docker(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    fake = FakeDocker()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def test_backend_defaults_to_docker_and_rejects_typos(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "")
    assert sandbox_backend() == "docker"
    monkeypatch.setenv("SANDBOX_BACKEND", "Modal")
    assert sandbox_backend() == "modal"
    monkeypatch.setenv("SANDBOX_BACKEND", "dokcer")
    with pytest.raises(ValueError):
        sandbox_backend()


def test_container_is_started_like_a_modal_sandbox(docker):
    env = Environment(image="python:3.12", deployment_timeout=900, modal_sandbox_kwargs={"encrypted_ports": [8000]})
    run = docker.last("run")
    assert ["--detach", "--rm", "--init"] == run[2:5]
    assert run[run.index("--publish") + 1] == "127.0.0.1::8000"
    assert run[-4:] == ["--entrypoint", "sleep", "python:3.12", "900"]
    assert (env.system, env.machine) == ("Linux", "x86_64")
    assert env.tunnel_url(8000) == "http://127.0.0.1:49153"
    with pytest.raises(ValueError):
        env.tunnel_url(9000)


def test_shell_and_argv_commands(docker):
    env = Environment(cwd="/testbed")
    docker.handlers["exec"] = lambda args: (3, "out\n", "err\n")
    assert env.execute("echo hi", env={"A": "1"}) == {
        "stdout": "out\n", "stderr": "err\n", "output": "out\nerr\n", "returncode": 3, "exception_info": "",
    }
    shell = docker.last("exec")
    assert shell[shell.index("--workdir") + 1] == "/testbed"
    assert shell[shell.index("--env") + 1] == "A=1"
    # No per-call timeout falls back to runtime_timeout, as on SWE-ReX.
    assert shell[-7:] == ["/usr/bin/timeout", "--kill-after", "5", "600", "/bin/sh", "-c", "echo hi"]

    env.execute(["echo", "hi"], shell=False, cwd="/tmp")
    argv = docker.last("exec")
    assert argv[argv.index("--workdir") + 1] == "/tmp"
    assert "/bin/sh" not in argv and argv[-2:] == ["echo", "hi"]


def test_timeout_is_reported_not_raised(docker):
    env = Environment()

    def slow(args):
        time.sleep(0.05)
        return 124, "", ""

    docker.handlers["exec"] = slow
    result = env.execute("sleep 100", timeout=0.01)
    assert result["returncode"] == -1
    assert result["extra"]["exception_type"] == "CommandTimeoutError"

    # A fast exit with timeout's status is a normal result, not a timeout.
    docker.handlers["exec"] = lambda args: (124, "", "")
    assert env.execute("exit 124", timeout=60)["returncode"] == 124


def test_dead_container_is_terminal_and_stop_is_idempotent(docker):
    env = Environment()
    docker.handlers["exec"] = lambda args: (1, "", "Error response from daemon: container is not running")
    docker.handlers["container"] = lambda args: (1, "", "No such container")
    with pytest.raises(RuntimeError, match="no longer running"):
        env.execute("true")
    assert not env.is_alive()

    docker.handlers["rm"] = lambda args: (1, "", "Error: No such container: x")
    env.stop()
    env.stop()
    assert len([call for call in docker.calls if call[1] == "rm"]) == 1


def test_testbed_context_is_the_tracked_source_with_pins_last(docker, monkeypatch):
    captured = {}

    def fake_build(dockerfile, files, name, force_build=False):
        captured.update(dockerfile=dockerfile, files=[target for _, target in files], name=name)
        return "assignment-sandbox:test"

    monkeypatch.setattr(image_module, "build_image", fake_build)
    task = Task.load(ROOT / "tasks" / "chess-terminal-move")
    built = image_module.build_testbed_image(task)

    assert built == DockerImage("assignment-sandbox:test")
    tracked = REAL_RUN(["git", "-C", str(task.source), "ls-files"], capture_output=True, text=True)
    assert sorted(captured["files"]) == sorted(tracked.stdout.split())
    assert captured["dockerfile"].rstrip().splitlines()[-1] == (
        "RUN pip install --no-cache-dir " + " ".join(task.pins)
    )
    assert captured["name"] == task.id


def test_builds_are_cached_by_content(docker, tmp_path):
    built = set()
    docker.handlers["image"] = lambda args: (0, "id", "") if args[-1] in built | {"base:1"} else (1, "", "missing")

    def build(args):
        built.add(args[args.index("-t") + 1])
        return 0, "", ""

    docker.handlers["build"] = build
    script = tmp_path / "tool.py"
    script.write_text("print(1)\n")
    image = DockerImage("base:1").add_local_file(str(script), "/opt/assignment/tool.py")

    first = image.build()
    assert first.startswith("assignment-sandbox:files-") and image.build() == first
    assert len([call for call in docker.calls if call[1] == "build"]) == 1

    script.write_text("print(2)\n")
    assert image.build() != first


def test_status_explains_a_missing_cli(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)
    assert docker_backend.docker_status() == (False, "the docker CLI is not installed or not on PATH")
