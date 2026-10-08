"""The docker backend against a real daemon: the Modal suites' checks, locally.

Skipped when no daemon answers. The first run builds the chess testbed image.
"""

import base64
import json
import time
from pathlib import Path

import pytest

from assignment.docker_backend import docker_status

AVAILABLE, DETAIL = docker_status()
pytestmark = [pytest.mark.docker, pytest.mark.skipif(not AVAILABLE, reason=f"docker unavailable: {DETAIL}")]

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "chess-terminal-move"


@pytest.fixture(scope="module", autouse=True)
def docker_backend():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SANDBOX_BACKEND", "docker")
        yield


@pytest.fixture(scope="module")
def env(docker_backend):
    from assignment.env import Environment

    env = Environment()
    yield env
    env.stop()
    assert not env.is_alive(), "the container outlived stop()"


def test_shell_argv_and_failing_commands(env):
    assert env.execute("echo 'hello, world'")["output"] == "hello, world\n"
    assert env.execute(["echo", "hello, world"], shell=False)["output"] == "hello, world\n"
    failed = env.execute("python -c 'print(\"hello, world\"); raise Exception(\"test exception\")'")
    assert failed["returncode"] != 0
    assert "hello, world\n" in failed["output"] and "test exception" in failed["output"]


def test_timeout_kills_the_command(env):
    started = time.monotonic()
    result = env.execute("sleep 30", timeout=1)
    assert result["returncode"] == -1 and time.monotonic() - started < 15
    sleeping = env.execute("grep -lx sleep /proc/[0-9]*/comm 2>/dev/null | wc -l")["output"].strip()
    assert sleeping == "1", "only the keep-alive sleep should remain"


def test_chess_app_is_playable_through_the_published_port(docker_backend):
    from assignment.chess_sandbox import ChessSandbox, IllegalMove

    with ChessSandbox(task=TASK) as sandbox:
        assert sandbox.server_url.startswith("http://127.0.0.1:")
        current = sandbox.state()
        assert current["turn"] == "white" and current["history"] == []
        with pytest.raises(IllegalMove):
            sandbox.play("e2e5")
        assert sandbox.reset()["history"] == []

        code = base64.b64encode(b"print('from the sandbox')").decode()
        ran = sandbox.execute(f"python /opt/assignment/sandbox_python.py 8000 {code}")
        assert ran["returncode"] == 0, ran["stderr"]
        assert json.loads(ran["stdout"]) == {"stdout": "from the sandbox\n", "stderr": "", "error": None}


def test_unpatched_testbed_fails_only_the_regression_test(docker_backend):
    from assignment.eval import EvaluationSpec, Task, evaluate

    spec = EvaluationSpec.load(TASK / "public_tests")
    report = evaluate(spec, task=Task.load(TASK))
    assert not report.error, report.error
    assert all(status.value == "PASSED" for status in report.pass_to_pass.values()), report.summary()
    assert all(status.value != "PASSED" for status in report.fail_to_pass.values()), report.summary()
