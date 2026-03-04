from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import FINISH_TOOL_NAME

_FINISH_DESCRIPTION = f"""Use the {FINISH_TOOL_NAME} tool to indicate that the task is complete.
Only call this when you are certain that you have completed the task and tested your work.
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=FINISH_TOOL_NAME,
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'A final summary message describing what was accomplished.',
                },
            },
        },
    ),
)
