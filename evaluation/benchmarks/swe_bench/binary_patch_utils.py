"""Utilities for handling binary files and patch generation in SWE-bench evaluation."""


def remove_binary_diffs(patch_text):
    """Remove binary file diffs from a git patch.

    Args:
        patch_text (str): The git patch text

    Returns:
        str: The cleaned patch text with binary diffs removed
    """
    lines = patch_text.splitlines()
    cleaned_lines = []
    block = []
    is_binary_block = False

    for line in lines:
        if line.startswith('diff --git '):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif 'Binary files' in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return '\n'.join(cleaned_lines)


def remove_binary_files_from_git():
    """Generate a bash command to remove binary files from git staging.

    Detects binary files by MIME encoding (the word `binary` from
    `file --mime-encoding`, which is libmagic's authoritative check) or
    by an explicit `binary: set` attribute. The previous `file | grep
    executable` check matched any libmagic description containing the
    word "executable", which falsely included every Python source file
    (libmagic reports plain .py as `Python script, ASCII text
    executable`) — that caused the agent's edited source files to be
    `git rm -f`'d and show up as `deleted file mode` in the final patch.

    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && ([ "$(file --mime-encoding -b "$file" 2>/dev/null)" = "binary" ] || git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()
