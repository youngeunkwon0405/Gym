from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import CODEX_APPLY_PATCH_TOOL_NAME

_APPLY_PATCH_DESCRIPTION = r"""Use the apply_patch tool to edit files.

It should be used whenever you want to create or edit a file.

IMPORTANT: NEVER, EVER use apply_patch to write tests, test files, or any code that verifies behavior. Only use it for production code, configurations, and documentation.

Patch format:

Each patch should be a series of *** Begin Patch / *** End Patch blocks, one per file:

*** Begin Patch
[file operation]
*** End Patch

Supported file operations:

1. Update a file:
*** Begin Patch
--- a/path/to/file
+++ b/path/to/file
@@ context line from original file @@
-line to remove
+line to add
 context line (unchanged)
*** End Patch

2. Create a new file:
*** Begin Patch
--- /dev/null
+++ b/path/to/new/file
+first line
+second line
*** End Patch

3. Delete a file:
*** Begin Patch
--- a/path/to/file
+++ /dev/null
*** End Patch

Notes:
- Context lines (starting with a space) must match exactly.
- The @@ line must contain a line that exists in the original file to anchor the patch.
- Multiple hunks can be included for the same file by using multiple @@ anchors.
- To move/rename a file, delete the old path and create the new path.
"""

ApplyPatchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=CODEX_APPLY_PATCH_TOOL_NAME,
        description=_APPLY_PATCH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['input'],
            'properties': {
                'input': {
                    'type': 'string',
                    'description': 'The entire contents of the apply_patch command, including the *** Begin Patch and *** End Patch delimiters.',
                },
            },
            'additionalProperties': False,
        },
    ),
)
