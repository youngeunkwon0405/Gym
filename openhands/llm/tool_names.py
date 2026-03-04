import os
import random

_DIVERSIFY = os.environ.get("DIVERSIFY_TOOL_NAMES", "false").lower() == "true"
_CAMEL_CASE_TOOL_NAMES = (
    os.environ.get("CAMEL_CASE_TOOL_NAMES", "false").lower() == "true"
)
_SEED = os.environ.get("TOOL_NAME_SEED")

if _DIVERSIFY:
    _rng = random.Random(int(_SEED) if _SEED is not None else None)
else:
    _rng = random.Random()


def _pick(default: str, alternatives: list[str]) -> str:
    choice = default

    if _DIVERSIFY:
        choice = _rng.choice(alternatives)
    if _CAMEL_CASE_TOOL_NAMES:
        choice = "".join(word.capitalize() for word in choice.split("_"))

    return choice


# ---------- CodeAct tools ----------
EXECUTE_BASH_TOOL_NAME = _pick(
    "execute_bash",
    [
        "execute_bash",
        "run_bash",
        "bash_exec",
        "shell_run",
        "shell",
    ],
)
STR_REPLACE_EDITOR_TOOL_NAME = _pick(
    "str_replace_editor",
    [
        "str_replace_editor",
        "text_editor",
        "file_editor",
        "code_editor",
    ],
)
BROWSER_TOOL_NAME = _pick(
    "browser",
    [
        "browser",
        "web_browser",
        "browse_web",
        "open_browser",
    ],
)
FINISH_TOOL_NAME = _pick(
    "finish",
    ["finish"],
)
LLM_BASED_EDIT_TOOL_NAME = _pick(
    "edit_file",
    [
        "edit_file",
        "modify_file",
        "update_file",
        "file_edit",
    ],
)
TASK_TRACKER_TOOL_NAME = _pick(
    "task_tracker",
    [
        "task_tracker",
        "todo_tracker",
        "plan_tracker",
    ],
)

# ---------- OpenCode-inspired tools ----------
BASH_TOOL_NAME = _pick(
    "bash",
    ["bash", "shell", "terminal"],
)
GLOB_TOOL_NAME = _pick(
    "glob",
    ["glob", "find_files", "file_glob", "match_files"],
)
GREP_TOOL_NAME = _pick(
    "grep",
    ["grep", "search_text", "find_pattern", "text_search"],
)
LIST_DIR_TOOL_NAME = _pick(
    "list_dir",
    ["list_dir", "ls", "show_dir"],
)
READ_TOOL_NAME = _pick(
    "read",
    ["read", "read_file", "view_file", "cat_file"],
)
WRITE_TOOL_NAME = _pick(
    "write",
    ["write", "write_file", "save_file", "create_file"],
)
EDIT_TOOL_NAME = _pick(
    "edit",
    ["edit", "modify", "edit_file"],
)

# ---------- OpenCode additional tools ----------
OPENCODE_APPLY_PATCH_TOOL_NAME = _pick(
    "apply_patch",
    ["apply_patch", "patch_file", "apply_diff", "apply_changes"],
)
QUESTION_TOOL_NAME = _pick(
    "question",
    ["question", "ask_user", "clarify", "prompt_user"],
)
TODO_READ_TOOL_NAME = _pick(
    "todo_read",
    ["todo_read", "read_todos", "list_tasks"],
)
TODO_WRITE_TOOL_NAME = _pick(
    "todo_write",
    ["todo_write", "write_todos", "update_tasks"],
)

# ---------- Readonly agent tools ----------
VIEW_TOOL_NAME = _pick(
    "view",
    ["view", "file_view", "inspect", "show_file"],
)

# ---------- Codex-inspired tools ----------
CODEX_SHELL_COMMAND_TOOL_NAME = _pick(
    "shell_command",
    [
        "shell_command",
        "shell_cmd",
        "run_shell_cmd",
        "exec_command",
        "exec_cmd",
    ],
)
CODEX_READ_FILE_TOOL_NAME = _pick(
    "read_file",
    [
        "read_file",
        "view_file",
        "cat_file",
        "read_file_content",
    ],
)
CODEX_LIST_DIR_TOOL_NAME = _pick(
    "list_dir",
    [
        "list_dir",
        "ls_dir",
        "list_directory",
    ],
)
CODEX_GREP_FILES_TOOL_NAME = _pick(
    "grep_files",
    [
        "grep_files",
        "search_files",
        "find_in_files",
        "code_search",
    ],
)
CODEX_APPLY_PATCH_TOOL_NAME = _pick(
    "apply_patch",
    [
        "apply_patch",
        "patch_file",
        "apply_diff",
        "apply_changes",
    ],
)
CODEX_UPDATE_PLAN_TOOL_NAME = _pick(
    "update_plan",
    [
        "update_plan",
        "modify_plan",
        "edit_plan",
    ],
)
