"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.1.a): Define an OpenAI function-tool schema named ``play_move``.
# It must accept exactly one required string argument named ``move``, explain
# that moves use UCI notation (for example e2e4), and reject extra arguments.
PLAY_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one move for White on the live game board. The server then "
            "plays Black's reply automatically and returns the new state.\n"
            "\n"
            "Moves use UCI notation: the from-square followed by the to-square, "
            "for example `e2e4` or `g1f3`. Castling is written as the king's "
            "move, such as `e1g1`, and a promotion appends the piece letter, "
            "such as `e7e8q`."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": (
                        "White's move in UCI notation, for example e2e4 or e7e8q."
                    ),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.3): Define the `simulate_move` tool, like the `play_move` tool.
SIMULATE_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Inspect a hypothetical position, or play one ply in it, without "
            "touching the live game. The real board does not change and the "
            "opponent does not reply.\n"
            "\n"
            "With `move` set to null, returns the position described by `fen` "
            "and its legal moves. With a UCI `move` (for example `e2e4` or "
            "`e7e8q`), applies that single move for whichever side is to move "
            "and returns the resulting position. The result is JSON with `fen`, "
            "`squares`, `turn`, `legal_moves`, `in_check`, `game_over`, "
            "`winner`, and `result`. Pass a returned `fen` back in to look "
            "further ahead."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "A complete six-field FEN of the position.",
                },
                "move": {
                    "type": ["string", "null"],
                    "description": (
                        "One UCI move to apply in that position, or null to "
                        "only inspect it."
                    ),
                },
            },
            "required": ["fen", "move"],
            "additionalProperties": False,
        },
    },
}

# TODO()
RUN_PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a Python snippet in the sandbox beside the chess server. "
            "Returns a JSON object with the snippet's captured `stdout`, "
            "`stderr`, and `error`, followed by the live board after it ran.\n"
            "\n"
            "Inside the snippet, `simulate_move(fen, move=None)` and "
            "`play_move(move)` are already defined as ordinary synchronous "
            "functions; do not import them. Both return the resulting state as "
            "a dict and raise an exception on an illegal move or bad input. "
            "`simulate_move` never changes the live game. `play_move` commits a "
            "real move for White, and Black replies. Use `print` to report what "
            "the snippet found."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to execute.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
