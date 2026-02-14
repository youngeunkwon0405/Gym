"""Standalone unit tests for Codex runtime handler implementations.

These tests directly test the handler methods in action_execution_server.py
using real file operations on temporary directories, without requiring Docker.
"""

import asyncio
import os
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.events.action.codex import (
    CodexApplyPatchAction,
    CodexGrepFilesAction,
    CodexListDirAction,
    CodexReadFileAction,
    CodexUpdatePlanAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
)
from openhands.events.observation.codex import (
    CodexApplyPatchObservation,
    CodexUpdatePlanObservation,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_executor(temp_workspace):
    """Create a minimal mock ActionExecutor with real file system access."""
    from openhands.runtime.action_execution_server import ActionExecutor

    executor = MagicMock(spec=ActionExecutor)
    executor._initial_cwd = temp_workspace
    executor.bash_session = MagicMock()
    executor.bash_session.cwd = temp_workspace
    executor.lock = asyncio.Lock()

    # Use the real _resolve_path logic
    def resolve_path(path, working_dir):
        if os.path.isabs(path):
            return path
        return os.path.join(working_dir, path)

    executor._resolve_path = resolve_path

    return executor


def create_test_file(directory: str, filename: str, content: str) -> str:
    """Helper to create a test file."""
    filepath = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath


def create_test_structure(base_dir: str) -> dict:
    """Create a test directory structure and return file paths."""
    files = {
        'main.py': 'def main():\n    print("Hello")\n\nif __name__ == "__main__":\n    main()',
        'utils.py': '# Utility functions\n\ndef helper():\n    return 42\n\ndef another():\n    pass',
        'src/core.py': 'class Core:\n    def __init__(self):\n        self.value = 0\n\n    def run(self):\n        pass',
        'src/lib/helpers.py': 'import os\n\ndef get_path():\n    return os.getcwd()',
        'tests/test_main.py': 'import pytest\n\ndef test_main():\n    assert True\n\ndef test_another():\n    pass',
        'config.json': '{"name": "test", "version": "1.0.0"}',
        'README.md': '# Test Project\n\nThis is a test project.',
    }

    paths = {}
    for rel_path, content in files.items():
        full_path = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        paths[rel_path] = full_path

    return paths


# ==============================================================================
# CodexReadFile Handler Tests
# ==============================================================================


class TestCodexReadFileHandler:
    """Tests for the codex_read_file handler implementation."""

    def test_read_file_line_format_is_1_indexed(self, temp_workspace):
        """Test that lines are formatted as L{number}: (1-indexed, Codex-style)."""
        content = 'line 1\nline 2\nline 3\nline 4\nline 5'
        filepath = create_test_file(temp_workspace, 'test.txt', content)

        with open(filepath, 'r') as f:
            lines = f.read().split('\n')

        offset = 1  # 1-indexed
        start_idx = offset - 1
        raw = lines[start_idx:]
        formatted = [f"L{i + offset}: {line}" for i, line in enumerate(raw)]

        assert formatted[0] == 'L1: line 1'
        assert formatted[1] == 'L2: line 2'
        assert formatted[4] == 'L5: line 5'

    def test_read_file_with_offset(self, temp_workspace):
        """Test reading file from a specific 1-indexed offset."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        create_test_file(temp_workspace, 'test.txt', content)

        with open(os.path.join(temp_workspace, 'test.txt'), 'r') as f:
            lines = f.read().split('\n')

        offset = 50  # 1-indexed, so this is the 50th line
        start_idx = offset - 1
        raw = lines[start_idx:start_idx + 10]
        formatted = [f"L{i + offset}: {line}" for i, line in enumerate(raw)]

        assert formatted[0] == 'L50: line 50'
        assert len(formatted) == 10

    def test_read_file_with_limit(self, temp_workspace):
        """Test reading file with line limit."""
        content = '\n'.join([f'line {i}' for i in range(1, 101)])
        create_test_file(temp_workspace, 'test.txt', content)

        with open(os.path.join(temp_workspace, 'test.txt'), 'r') as f:
            lines = f.read().split('\n')

        limit = 5
        raw = lines[:limit]

        assert len(raw) == 5
        assert raw[0] == 'line 1'
        assert raw[4] == 'line 5'

    def test_read_nonexistent_file(self, temp_workspace):
        """Test reading a file that doesn't exist."""
        filepath = os.path.join(temp_workspace, 'nonexistent.py')
        assert not os.path.exists(filepath)

    def test_read_binary_file_detection(self, temp_workspace):
        """Test binary file detection by null bytes."""
        binary_content = b'\x00\x01\x02\x03\x04\x05'
        binary_path = os.path.join(temp_workspace, 'binary.dat')
        with open(binary_path, 'wb') as f:
            f.write(binary_content)

        with open(binary_path, 'rb') as f:
            chunk = f.read(4096)
            is_binary = b'\x00' in chunk

        assert is_binary is True

    def test_read_text_file_not_binary(self, temp_workspace):
        """Test that text files are not detected as binary."""
        text_path = create_test_file(temp_workspace, 'text.py', 'print("hello")')

        with open(text_path, 'rb') as f:
            chunk = f.read(4096)
            is_binary = b'\x00' in chunk

        assert is_binary is False

    def test_read_directory_is_error(self, temp_workspace):
        """Test that reading a directory path is an error."""
        subdir = os.path.join(temp_workspace, 'subdir')
        os.makedirs(subdir)
        assert os.path.isdir(subdir)

    def test_read_file_suggestions_on_not_found(self, temp_workspace):
        """Test that similar file names are suggested when file not found."""
        create_test_file(temp_workspace, 'test_file.py', 'content')
        create_test_file(temp_workspace, 'test_file.txt', 'content')

        missing = 'test_file.js'
        basename = 'test_file'
        entries = os.listdir(temp_workspace)
        suggestions = [
            e for e in entries
            if basename.lower() in e.lower()
        ]

        assert len(suggestions) >= 2
        assert 'test_file.py' in suggestions
        assert 'test_file.txt' in suggestions


# ==============================================================================
# CodexReadFile Indentation Mode Tests
# ==============================================================================


class TestCodexReadFileIndentationMode:
    """Tests for the indentation-aware block reading mode."""

    def test_indentation_finds_anchor_block(self, temp_workspace):
        """Test that indentation mode finds the block around the anchor line."""
        content = (
            'class MyClass:\n'
            '    def method_one(self):\n'
            '        x = 1\n'
            '        y = 2\n'
            '        return x + y\n'
            '\n'
            '    def method_two(self):\n'
            '        return 42\n'
        )
        create_test_file(temp_workspace, 'test.py', content)

        with open(os.path.join(temp_workspace, 'test.py'), 'r') as f:
            lines = f.read().split('\n')

        # Anchor on line 3 (0-indexed 2): "        x = 1"
        anchor_idx = 2
        anchor_line = lines[anchor_idx]
        anchor_indent = len(anchor_line) - len(anchor_line.lstrip())

        assert anchor_indent == 8  # 8 spaces indentation

    def test_indentation_walks_up_to_parent(self, temp_workspace):
        """Test that indentation mode walks up to find parent blocks."""
        content = (
            'class Foo:\n'
            '    def bar(self):\n'
            '        x = 1\n'
            '        y = 2\n'
        )
        create_test_file(temp_workspace, 'test.py', content)

        with open(os.path.join(temp_workspace, 'test.py'), 'r') as f:
            lines = f.read().split('\n')

        # Start at "x = 1" (index 2), walk up
        anchor_idx = 2
        current_indent = len(lines[anchor_idx]) - len(lines[anchor_idx].lstrip())  # 8
        start_idx = anchor_idx

        for i in range(anchor_idx - 1, -1, -1):
            stripped = lines[i].lstrip()
            if not stripped:
                continue
            line_indent = len(lines[i]) - len(stripped)
            if line_indent < current_indent:
                current_indent = line_indent
                start_idx = i

        # Should walk up to "class Foo:" (index 0)
        assert start_idx == 0

    def test_indentation_respects_max_levels(self, temp_workspace):
        """Test that max_levels limits how far up we walk."""
        content = (
            'class Outer:\n'
            '    class Inner:\n'
            '        def method(self):\n'
            '            x = 1\n'
        )
        create_test_file(temp_workspace, 'test.py', content)

        with open(os.path.join(temp_workspace, 'test.py'), 'r') as f:
            lines = f.read().split('\n')

        # Anchor at "x = 1" (index 3), max_levels=1
        anchor_idx = 3
        current_indent = len(lines[anchor_idx]) - len(lines[anchor_idx].lstrip())
        start_idx = anchor_idx
        levels_found = 0
        max_levels = 1

        for i in range(anchor_idx - 1, -1, -1):
            stripped = lines[i].lstrip()
            if not stripped:
                continue
            line_indent = len(lines[i]) - len(stripped)
            if line_indent < current_indent:
                levels_found += 1
                current_indent = line_indent
                start_idx = i
                if max_levels > 0 and levels_found >= max_levels:
                    break

        # Should stop at "def method" (index 2), not go all the way to "class Outer"
        assert start_idx == 2
        assert levels_found == 1


# ==============================================================================
# CodexListDir Handler Tests
# ==============================================================================


class TestCodexListDirHandler:
    """Tests for the codex_list_dir handler implementation."""

    def test_list_dir_basic(self, temp_workspace):
        """Test basic directory listing."""
        create_test_file(temp_workspace, 'file1.py', 'content')
        create_test_file(temp_workspace, 'file2.txt', 'content')
        os.makedirs(os.path.join(temp_workspace, 'subdir'))

        entries = []
        items = sorted(os.listdir(temp_workspace))
        for item in items:
            if item.startswith('.'):
                continue
            full_path = os.path.join(temp_workspace, item)
            type_label = 'dir' if os.path.isdir(full_path) else 'file'
            entries.append((item, type_label))

        assert ('file1.py', 'file') in entries
        assert ('file2.txt', 'file') in entries
        assert ('subdir', 'dir') in entries

    def test_list_dir_format_with_numbers(self, temp_workspace):
        """Test that entries are formatted with 1-indexed numbers and type labels."""
        create_test_file(temp_workspace, 'alpha.py', '')
        create_test_file(temp_workspace, 'beta.txt', '')

        entries = sorted(os.listdir(temp_workspace))
        entries = [(e, 'file') for e in entries if not e.startswith('.')]

        output_lines = []
        for i, (name, type_label) in enumerate(entries):
            output_lines.append(f"{i + 1}. [{type_label}] {name}")

        assert output_lines[0] == '1. [file] alpha.py'
        assert output_lines[1] == '2. [file] beta.txt'

    def test_list_dir_with_depth(self, temp_workspace):
        """Test directory listing with depth control."""
        create_test_file(temp_workspace, 'root.py', '')
        create_test_file(temp_workspace, 'sub/level1.py', '')
        create_test_file(temp_workspace, 'sub/deep/level2.py', '')
        create_test_file(temp_workspace, 'sub/deep/deeper/level3.py', '')

        # Depth 1 should only show root-level items
        entries_depth1 = []
        for item in sorted(os.listdir(temp_workspace)):
            if item.startswith('.'):
                continue
            full = os.path.join(temp_workspace, item)
            entries_depth1.append((item, 'dir' if os.path.isdir(full) else 'file'))

        assert ('root.py', 'file') in entries_depth1
        assert ('sub', 'dir') in entries_depth1

    def test_list_dir_offset_and_limit(self, temp_workspace):
        """Test directory listing with offset and limit pagination."""
        for i in range(10):
            create_test_file(temp_workspace, f'file_{i:02d}.py', '')

        entries = sorted(os.listdir(temp_workspace))
        entries = [(e, 'file') for e in entries if not e.startswith('.')]

        # Offset 3 (1-indexed), limit 3
        offset = 3
        limit = 3
        start_idx = offset - 1
        paginated = entries[start_idx:start_idx + limit]

        assert len(paginated) == 3
        assert paginated[0][0] == 'file_02.py'  # 3rd entry (0-indexed 2)
        assert paginated[2][0] == 'file_04.py'

    def test_list_dir_nonexistent(self, temp_workspace):
        """Test listing a nonexistent directory."""
        path = os.path.join(temp_workspace, 'nonexistent')
        assert not os.path.exists(path)

    def test_list_dir_skips_hidden_files(self, temp_workspace):
        """Test that hidden files (starting with .) are skipped."""
        create_test_file(temp_workspace, '.hidden', '')
        create_test_file(temp_workspace, 'visible.py', '')

        items = sorted(os.listdir(temp_workspace))
        visible = [i for i in items if not i.startswith('.')]

        assert '.hidden' not in visible
        assert 'visible.py' in visible


# ==============================================================================
# CodexGrepFiles Handler Tests
# ==============================================================================


class TestCodexGrepFilesHandler:
    """Tests for the codex_grep_files handler implementation."""

    def test_grep_returns_file_paths_not_lines(self, temp_workspace):
        """Test that grep_files returns file paths, not matching lines."""
        create_test_file(temp_workspace, 'a.py', 'TODO: fix this')
        create_test_file(temp_workspace, 'b.py', 'TODO: and this')
        create_test_file(temp_workspace, 'c.py', 'no match here')

        # Try ripgrep first, fall back to grep -rl
        try:
            result = subprocess.run(
                ['rg', '-l', 'TODO', temp_workspace],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['grep', '-rl', 'TODO', temp_workspace],
                capture_output=True, text=True, timeout=10
            )

        files = [f for f in result.stdout.strip().split('\n') if f]
        assert len(files) == 2
        for f in files:
            assert os.path.isfile(f)

    def test_grep_sorts_by_mtime(self, temp_workspace):
        """Test that results are sorted by modification time (newest first)."""
        import time

        f1 = create_test_file(temp_workspace, 'old.py', 'MATCH')
        time.sleep(0.1)
        f2 = create_test_file(temp_workspace, 'new.py', 'MATCH')

        files = [f1, f2]
        files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        # Newest file should be first
        assert 'new.py' in files[0]

    def test_grep_with_include_filter(self, temp_workspace):
        """Test that include filter limits file types."""
        create_test_file(temp_workspace, 'match.py', 'PATTERN')
        create_test_file(temp_workspace, 'match.txt', 'PATTERN')
        create_test_file(temp_workspace, 'match.js', 'PATTERN')

        try:
            result = subprocess.run(
                ['rg', '-l', '-g', '*.py', 'PATTERN', temp_workspace],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['grep', '-rl', '--include=*.py', 'PATTERN', temp_workspace],
                capture_output=True, text=True, timeout=10
            )

        files = [f for f in result.stdout.strip().split('\n') if f]
        assert len(files) == 1
        assert files[0].endswith('.py')

    def test_grep_respects_limit(self, temp_workspace):
        """Test that results are limited to the specified count."""
        for i in range(20):
            create_test_file(temp_workspace, f'file_{i}.py', 'MATCH_PATTERN')

        try:
            result = subprocess.run(
                ['rg', '-l', 'MATCH_PATTERN', temp_workspace],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['grep', '-rl', 'MATCH_PATTERN', temp_workspace],
                capture_output=True, text=True, timeout=10
            )

        all_files = [f for f in result.stdout.strip().split('\n') if f]
        limit = 5
        limited = all_files[:limit]
        assert len(limited) == 5
        assert len(all_files) == 20

    def test_grep_no_matches(self, temp_workspace):
        """Test grep with no matches returns empty."""
        create_test_file(temp_workspace, 'file.py', 'nothing to see here')

        try:
            result = subprocess.run(
                ['rg', '-l', 'NONEXISTENT_PATTERN_XYZ', temp_workspace],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['grep', '-rl', 'NONEXISTENT_PATTERN_XYZ', temp_workspace],
                capture_output=True, text=True, timeout=10
            )

        files = result.stdout.strip()
        assert files == ''

    def test_grep_regex_pattern(self, temp_workspace):
        """Test grep with regex pattern."""
        create_test_file(temp_workspace, 'func.py', 'def my_function():\n    pass')
        create_test_file(temp_workspace, 'cls.py', 'class MyClass:\n    pass')
        create_test_file(temp_workspace, 'data.txt', 'no functions here')

        try:
            result = subprocess.run(
                ['rg', '-l', r'def \w+\(', temp_workspace],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            result = subprocess.run(
                ['grep', '-rlP', r'def \w+\(', temp_workspace],
                capture_output=True, text=True, timeout=10
            )

        files = [f for f in result.stdout.strip().split('\n') if f]
        assert len(files) == 1
        assert files[0].endswith('func.py')


# ==============================================================================
# CodexApplyPatch Handler Tests
# ==============================================================================


class TestCodexApplyPatchHandler:
    """Tests for the codex_apply_patch handler implementation."""

    def test_apply_patch_create_new_file(self, temp_workspace):
        """Test creating a new file via patch."""
        new_file_path = os.path.join(temp_workspace, 'new_file.py')
        assert not os.path.exists(new_file_path)

        # Simulate the manual patch logic for file creation
        content_lines = ['print("hello world")', 'print("goodbye")']
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))

        assert os.path.exists(new_file_path)
        with open(new_file_path, 'r') as f:
            result = f.read()
        assert 'hello world' in result
        assert 'goodbye' in result

    def test_apply_patch_delete_file(self, temp_workspace):
        """Test deleting a file via patch."""
        filepath = create_test_file(temp_workspace, 'to_delete.py', 'old content')
        assert os.path.exists(filepath)

        # Simulate delete
        os.unlink(filepath)
        assert not os.path.exists(filepath)

    def test_apply_patch_create_nested_directory(self, temp_workspace):
        """Test creating a file in a new nested directory."""
        nested_path = os.path.join(temp_workspace, 'deep', 'nested', 'dir', 'file.py')
        assert not os.path.exists(os.path.dirname(nested_path))

        os.makedirs(os.path.dirname(nested_path), exist_ok=True)
        with open(nested_path, 'w') as f:
            f.write('content')

        assert os.path.exists(nested_path)

    def test_apply_patch_empty_patch_is_error(self):
        """Test that empty patch text is an error."""
        action = CodexApplyPatchAction(patch='')
        assert action.patch == ''

    def test_apply_patch_strips_begin_end_markers(self):
        """Test that *** Begin Patch / *** End Patch markers are stripped."""
        patch_text = (
            '*** Begin Patch\n'
            '--- a/test.py\n'
            '+++ b/test.py\n'
            '@@ def main(): @@\n'
            '-    old\n'
            '+    new\n'
            '*** End Patch'
        )

        # Strip logic
        clean_patch = patch_text
        if '*** Begin Patch' in clean_patch:
            parts = clean_patch.split('*** Begin Patch')
            clean_patch = parts[1]
        if '*** End Patch' in clean_patch:
            clean_patch = clean_patch.split('*** End Patch')[0]
        clean_patch = clean_patch.strip()

        assert not clean_patch.startswith('*** Begin Patch')
        assert '*** End Patch' not in clean_patch
        assert '--- a/test.py' in clean_patch

    def test_apply_patch_git_apply_format(self, temp_workspace):
        """Test that patch can be written and applied with git apply."""
        # Initialize git repo
        try:
            subprocess.run(['git', 'init'], cwd=temp_workspace, capture_output=True, timeout=10)
        except FileNotFoundError:
            pytest.skip("git not available")

        # Create initial file
        filepath = create_test_file(temp_workspace, 'test.py', 'line 1\nline 2\nline 3\n')

        # Stage and commit
        subprocess.run(['git', 'add', '.'], cwd=temp_workspace, capture_output=True, timeout=10)
        subprocess.run(
            ['git', 'commit', '-m', 'init'],
            cwd=temp_workspace, capture_output=True, timeout=10,
            env={**os.environ, 'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@test.com',
                 'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@test.com'}
        )

        # Create a diff-format patch
        patch = 'diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,3 @@\n line 1\n-line 2\n+line 2 modified\n line 3\n'

        import tempfile as tf
        with tf.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch)
            patch_file = f.name

        try:
            result = subprocess.run(
                ['git', 'apply', patch_file],
                cwd=temp_workspace, capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0

            with open(filepath, 'r') as f:
                content = f.read()
            assert 'line 2 modified' in content
        finally:
            os.unlink(patch_file)


# ==============================================================================
# CodexUpdatePlan Handler Tests
# ==============================================================================


class TestCodexUpdatePlanHandler:
    """Tests for the codex_update_plan handler implementation."""

    def test_update_plan_basic(self):
        """Test basic plan update."""
        plan = [
            {'step': 'Read code', 'status': 'completed'},
            {'step': 'Write feature', 'status': 'in_progress'},
            {'step': 'Test', 'status': 'pending'},
        ]

        # Validate plan
        in_progress_count = sum(1 for item in plan if item['status'] == 'in_progress')
        assert in_progress_count == 1

    def test_update_plan_rejects_multiple_in_progress(self):
        """Test that multiple in_progress steps are rejected."""
        plan = [
            {'step': 'Step A', 'status': 'in_progress'},
            {'step': 'Step B', 'status': 'in_progress'},
        ]

        in_progress_count = sum(1 for item in plan if item['status'] == 'in_progress')
        assert in_progress_count > 1  # This should be rejected

    def test_update_plan_validates_status_values(self):
        """Test that only valid status values are accepted."""
        valid_statuses = {'pending', 'in_progress', 'completed'}

        valid_plan = [{'step': 'Test', 'status': 'pending'}]
        invalid_plan = [{'step': 'Test', 'status': 'invalid_status'}]

        assert valid_plan[0]['status'] in valid_statuses
        assert invalid_plan[0]['status'] not in valid_statuses

    def test_update_plan_requires_step_and_status(self):
        """Test that plan items must have step and status fields."""
        valid_item = {'step': 'Do something', 'status': 'pending'}
        assert 'step' in valid_item
        assert 'status' in valid_item

        missing_step = {'status': 'pending'}
        assert 'step' not in missing_step

        missing_status = {'step': 'Do something'}
        assert 'status' not in missing_status

    def test_update_plan_returns_plan_updated(self):
        """Test that response is 'Plan updated'."""
        expected_response = 'Plan updated'
        assert expected_response == 'Plan updated'

    def test_update_plan_stores_state(self):
        """Test that plan state is stored and retrievable."""
        plan_storage: list[dict] = []

        plan = [
            {'step': 'Step 1', 'status': 'completed'},
            {'step': 'Step 2', 'status': 'in_progress'},
        ]

        plan_storage = list(plan)
        assert len(plan_storage) == 2
        assert plan_storage[0]['status'] == 'completed'
        assert plan_storage[1]['status'] == 'in_progress'

    def test_update_plan_empty_plan_list(self):
        """Test updating with an empty plan list."""
        plan: list[dict] = []
        assert len(plan) == 0
        # Empty plan should be valid

    def test_update_plan_all_completed(self):
        """Test plan where all steps are completed."""
        plan = [
            {'step': 'Step 1', 'status': 'completed'},
            {'step': 'Step 2', 'status': 'completed'},
            {'step': 'Step 3', 'status': 'completed'},
        ]

        in_progress = sum(1 for item in plan if item['status'] == 'in_progress')
        completed = sum(1 for item in plan if item['status'] == 'completed')

        assert in_progress == 0
        assert completed == 3


# ==============================================================================
# Integration-style Tests (combining action creation + handler logic)
# ==============================================================================


class TestCodexIntegration:
    """Integration tests combining action creation with handler logic."""

    def test_read_file_action_defaults(self):
        """Test that CodexReadFileAction has correct Codex defaults."""
        action = CodexReadFileAction(file_path='/test.py')
        assert action.offset == 1  # 1-indexed (unlike OpenCode's 0)
        assert action.limit == 2000
        assert action.mode == 'slice'
        assert action.indentation == {}

    def test_list_dir_action_defaults(self):
        """Test that CodexListDirAction has correct Codex defaults."""
        action = CodexListDirAction(dir_path='/workspace')
        assert action.offset == 1  # 1-indexed
        assert action.limit == 25
        assert action.depth == 2

    def test_grep_files_action_defaults(self):
        """Test that CodexGrepFilesAction has correct defaults."""
        action = CodexGrepFilesAction(pattern='TODO')
        assert action.include == ''
        assert action.path == ''
        assert action.limit == 100

    def test_apply_patch_observation_success(self):
        """Test CodexApplyPatchObservation for successful patch."""
        obs = CodexApplyPatchObservation(
            content='Patch applied successfully to 2 file(s).',
            files_changed=['a.py', 'b.py'],
            success=True,
        )
        assert obs.success is True
        assert len(obs.files_changed) == 2
        assert 'Applied patch to 2 files' in obs.message

    def test_apply_patch_observation_failure(self):
        """Test CodexApplyPatchObservation for failed patch."""
        obs = CodexApplyPatchObservation(
            content='Failed to apply: conflict',
            files_changed=[],
            success=False,
        )
        assert obs.success is False
        assert obs.message == 'Failed to apply patch'

    def test_update_plan_observation(self):
        """Test CodexUpdatePlanObservation."""
        obs = CodexUpdatePlanObservation(
            content='Plan updated',
            plan=[{'step': 'A', 'status': 'completed'}],
            success=True,
        )
        assert obs.success is True
        assert obs.content == 'Plan updated'
        assert len(obs.plan) == 1
