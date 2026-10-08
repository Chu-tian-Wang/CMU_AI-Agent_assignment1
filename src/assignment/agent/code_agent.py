"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

CODE_AGENT_SYSTEM_PROMPT = """You are an autonomous software engineering agent \
working in a sandboxed Linux terminal. You resolve issues in a code repository \
by acting through tool calls.

<system_information>
{system_information}
</system_information>

The repository is checked out at `{cwd}`, which is the default working \
directory for commands.

Work in short, deliberate steps:
1. Explore the repository to find the code relevant to the issue.
2. Reproduce the problem with a small script or test before changing anything.
3. Make a minimal, focused fix in the source files.
4. Re-run your reproduction and the relevant existing tests to verify the fix \
and check for regressions.

Rules:
- Every response must call a tool. Use `execute` to run bash commands, and \
`send_message` only once the task is completely finished; it ends the task.
- Each command runs in a fresh, non-interactive subshell, so `cd` and exports \
do not persist. Use the `cwd` and `env` arguments instead.
- Keep command output small: page through files with `sed -n`, `head`, or \
`grep -n` rather than printing whole files.
- Do not edit tests to make them pass; fix the underlying source code.
- Do not commit your changes."""

SKILLS_PROMPT = """Reusable skills are available. Each skill is a short guide \
for one kind of work. Call `invoke_skill` with a skill's name to load its full \
instructions before doing the work it covers, and follow them in place of your \
default approach.

<skills>
{catalog}
</skills>"""

TASK_PROMPT = """Resolve the following issue in the repository at `{cwd}`.

<issue>
{task}
</issue>"""

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = json.dumps(
            {
                "machine": environment.machine,
                "release": environment.release,
                "system": environment.system,
                "version": environment.version,
            },
            indent=2,
        )
        cwd = getattr(environment, "cwd", "/")
        self.system_prompt = CODE_AGENT_SYSTEM_PROMPT.format(
            system_information=system_information, cwd=cwd
        )
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            catalog = "\n\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += "\n\n" + SKILLS_PROMPT.format(catalog=catalog)
        self.task_prompt = TASK_PROMPT.format(cwd=cwd, task=task)

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = call.get("id", "")
            if self.finished:
                # Nothing may run after submission, but every call in the
                # action still needs a linked observation.
                content = format_tool_output(
                    {"error": "Not executed: the task was already submitted."}
                )
            else:
                function = call.get("function") or {}
                content = self._execute_one(
                    function.get("name", ""), function.get("arguments")
                )
            observations.append(
                {"role": "tool", "tool_call_id": call_id, "content": content}
            )
        return observations

    def _execute_one(self, name: str, raw_arguments: Any) -> str:
        """Run one tool call and return its observation text, never raising."""

        available = [tool["function"]["name"] for tool in self.tools]
        if name not in available:
            return format_tool_output(
                {
                    "error": (
                        f"Unknown tool {name!r}. Available tools: "
                        f"{', '.join(available)}."
                    )
                }
            )
        try:
            arguments = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            return format_tool_output(
                {"error": f"Tool arguments are not valid JSON ({exc}). Retry the call."}
            )
        if not isinstance(arguments, dict):
            return format_tool_output(
                {"error": "Tool arguments must be a JSON object. Retry the call."}
            )

        if name == "execute":
            return self._execute_command(arguments)
        if name == "send_message":
            if not isinstance(arguments.get("summary"), str):
                return format_tool_output(
                    {"error": "send_message needs a string `summary` argument."}
                )
            self.finished = True
            return format_tool_output({"status": "Message sent. The task is finished."})
        if name == "invoke_skill":
            skill_name = arguments.get("name")
            if not isinstance(skill_name, str) or skill_name not in self.skills:
                return format_tool_output(
                    {
                        "error": (
                            f"Unknown skill {skill_name!r}. Available skills: "
                            f"{', '.join(self.skills)}."
                        )
                    }
                )
            return self.skills[skill_name]["content"]
        return format_tool_output({"error": f"Tool {name!r} has no executor."})

    def _execute_command(self, arguments: dict[str, Any]) -> str:
        """Validate ``execute`` arguments and run the command in the sandbox."""

        command = arguments.get("command")
        if not (
            isinstance(command, str)
            or (isinstance(command, list) and all(isinstance(a, str) for a in command))
        ):
            return format_tool_output(
                {"error": "`command` must be a string or a list of strings."}
            )

        # Forward only the options the model set, so the environment's
        # defaults apply otherwise.
        options: dict[str, Any] = {}
        for key, expected in (
            ("shell", bool),
            ("cwd", str),
            ("timeout", (int, float)),
            ("env", dict),
        ):
            value = arguments.get(key)
            if value is None:
                continue
            if not isinstance(value, expected) or (
                key == "timeout" and isinstance(value, bool)
            ):
                return format_tool_output({"error": f"Invalid type for `{key}`."})
            options[key] = value
        if "env" in options:
            options["env"] = {str(k): str(v) for k, v in options["env"].items()}

        try:
            result = self.env.execute(command, **options)
        except Exception as exc:
            return format_tool_output(
                {"error": f"Command could not be executed: {type(exc).__name__}: {exc}"}
            )

        observation = {
            "output": result.get("output", ""),
            "returncode": result.get("returncode", ""),
        }
        if result.get("exception_info"):
            observation["exception_info"] = result["exception_info"]
        return format_tool_output(observation)
