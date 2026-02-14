from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME

_SHELL_COMMAND_DESCRIPTION = """Runs a shell command and returns its output.
- Always set the `workdir` param when using the shell_command function. Do not use `cd` unless absolutely necessary."""

ShellCommandTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_SHELL_COMMAND_TOOL_NAME,
        description=_SHELL_COMMAND_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['command'],
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The shell script to execute in the user\'s default shell',
                },
                'workdir': {
                    'type': 'string',
                    'description': 'The working directory to execute the command in',
                },
                'login': {
                    'type': 'boolean',
                    'description': 'Whether to run the shell with login shell semantics. Defaults to true.',
                },
                'timeout_ms': {
                    'type': 'number',
                    'description': 'The timeout for the command in milliseconds',
                },
            },
            'additionalProperties': False,
        },
    ),
)
