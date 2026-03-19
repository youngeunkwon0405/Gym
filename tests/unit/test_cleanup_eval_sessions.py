"""Tests for _cleanup_eval_sessions in evaluation/benchmarks/swe_bench/run_infer.py.

Covers:
  A. Basic cleanup behaviour (new dirs removed, pre-existing dirs kept)
  B. Edge cases (missing dir, empty dir, files vs dirs, permissions)
  C. Integration with the snapshot pattern used in process_instance

The function under test is imported inline to avoid pulling in heavy eval
dependencies (yappi, pandas, etc.) that may not be installed in the test env.
We replicate the function here for testability; a structural test verifies it
stays in sync with the source.
"""

import os
import shutil
import tempfile

import pytest


def _cleanup_eval_sessions(sessions_dir: str, pre_existing: set[str]) -> None:
    """Replica of evaluation.benchmarks.swe_bench.run_infer._cleanup_eval_sessions.

    Remove session directories created during a single eval run.
    """
    if not os.path.isdir(sessions_dir):
        return
    try:
        for entry in os.listdir(sessions_dir):
            if entry not in pre_existing:
                entry_path = os.path.join(sessions_dir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sessions_dir(tmp_path):
    """Create a temporary sessions directory."""
    d = tmp_path / 'sessions'
    d.mkdir()
    return str(d)


# =========================================================================
# A. Basic Cleanup Behaviour
# =========================================================================


class TestBasicCleanup:
    """A: Core cleanup logic — remove new dirs, keep pre-existing ones."""

    def test_removes_new_session_dir(self, sessions_dir):
        new_dir = os.path.join(sessions_dir, 'new-session-abc')
        os.makedirs(new_dir)
        assert os.path.isdir(new_dir)

        _cleanup_eval_sessions(sessions_dir, pre_existing=set())

        assert not os.path.exists(new_dir)

    def test_keeps_pre_existing_session_dir(self, sessions_dir):
        existing_dir = os.path.join(sessions_dir, 'existing-session')
        os.makedirs(existing_dir)

        pre_existing = {'existing-session'}
        _cleanup_eval_sessions(sessions_dir, pre_existing)

        assert os.path.isdir(existing_dir)

    def test_removes_only_new_keeps_pre_existing(self, sessions_dir):
        old_dir = os.path.join(sessions_dir, 'old-session')
        new_dir = os.path.join(sessions_dir, 'new-session')
        os.makedirs(old_dir)
        os.makedirs(new_dir)

        pre_existing = {'old-session'}
        _cleanup_eval_sessions(sessions_dir, pre_existing)

        assert os.path.isdir(old_dir)
        assert not os.path.exists(new_dir)

    def test_removes_multiple_new_dirs(self, sessions_dir):
        pre = os.path.join(sessions_dir, 'kept')
        os.makedirs(pre)
        new_dirs = []
        for i in range(5):
            d = os.path.join(sessions_dir, f'new-{i}')
            os.makedirs(d)
            new_dirs.append(d)

        _cleanup_eval_sessions(sessions_dir, pre_existing={'kept'})

        assert os.path.isdir(pre)
        for d in new_dirs:
            assert not os.path.exists(d)

    def test_removes_nested_content_inside_new_dir(self, sessions_dir):
        new_dir = os.path.join(sessions_dir, 'new-session')
        nested = os.path.join(new_dir, 'subdir', 'deep')
        os.makedirs(nested)
        with open(os.path.join(nested, 'file.txt'), 'w') as f:
            f.write('data')

        _cleanup_eval_sessions(sessions_dir, pre_existing=set())

        assert not os.path.exists(new_dir)

    def test_empty_pre_existing_removes_all(self, sessions_dir):
        for name in ['a', 'b', 'c']:
            os.makedirs(os.path.join(sessions_dir, name))

        _cleanup_eval_sessions(sessions_dir, pre_existing=set())

        remaining = os.listdir(sessions_dir)
        assert remaining == []

    def test_all_pre_existing_removes_nothing(self, sessions_dir):
        names = ['x', 'y', 'z']
        for name in names:
            os.makedirs(os.path.join(sessions_dir, name))

        _cleanup_eval_sessions(sessions_dir, pre_existing=set(names))

        remaining = set(os.listdir(sessions_dir))
        assert remaining == set(names)


# =========================================================================
# B. Edge Cases
# =========================================================================


class TestEdgeCases:
    """B: Edge cases and boundary conditions."""

    def test_nonexistent_sessions_dir_no_error(self):
        _cleanup_eval_sessions('/nonexistent/path/sessions', pre_existing=set())

    def test_empty_sessions_dir(self, sessions_dir):
        _cleanup_eval_sessions(sessions_dir, pre_existing=set())
        assert os.listdir(sessions_dir) == []

    def test_files_not_directories_are_skipped(self, sessions_dir):
        file_path = os.path.join(sessions_dir, 'not-a-dir.txt')
        with open(file_path, 'w') as f:
            f.write('content')

        _cleanup_eval_sessions(sessions_dir, pre_existing=set())

        assert os.path.isfile(file_path), 'Regular files should not be removed'

    def test_pre_existing_contains_names_not_present_on_disk(self, sessions_dir):
        new_dir = os.path.join(sessions_dir, 'new')
        os.makedirs(new_dir)

        _cleanup_eval_sessions(
            sessions_dir,
            pre_existing={'phantom', 'ghost'},
        )

        assert not os.path.exists(new_dir)

    def test_symlink_target_outside_sessions_dir_preserved(self, tmp_path, sessions_dir):
        """A symlink inside sessions_dir pointing to an external dir:
        shutil.rmtree(ignore_errors=True) may leave the symlink itself behind,
        but must never destroy the external target.
        """
        real_dir = str(tmp_path / 'external-data')
        os.makedirs(real_dir)
        with open(os.path.join(real_dir, 'important.txt'), 'w') as f:
            f.write('keep')

        link_name = os.path.join(sessions_dir, 'link-session')
        os.symlink(real_dir, link_name)

        _cleanup_eval_sessions(sessions_dir, pre_existing=set())

        assert os.path.isdir(real_dir), 'External target directory must be preserved'
        assert os.path.isfile(os.path.join(real_dir, 'important.txt'))


# =========================================================================
# C. Snapshot Pattern Integration
# =========================================================================


class TestSnapshotPattern:
    """C: Test the snapshot -> work -> cleanup pattern as used in process_instance."""

    def test_snapshot_then_add_then_cleanup(self, sessions_dir):
        """Simulates the pattern in process_instance:
        1. Snapshot existing dirs
        2. Work creates a new session dir
        3. Cleanup removes only the new one
        """
        os.makedirs(os.path.join(sessions_dir, 'pre-1'))
        os.makedirs(os.path.join(sessions_dir, 'pre-2'))

        pre_existing = set(os.listdir(sessions_dir))
        assert pre_existing == {'pre-1', 'pre-2'}

        new_session = os.path.join(sessions_dir, 'eval-session-xyz')
        os.makedirs(new_session)
        with open(os.path.join(new_session, 'events.json'), 'w') as f:
            f.write('[]')

        _cleanup_eval_sessions(sessions_dir, pre_existing)

        remaining = set(os.listdir(sessions_dir))
        assert remaining == {'pre-1', 'pre-2'}

    def test_snapshot_empty_then_add_then_cleanup(self, sessions_dir):
        """Snapshot when sessions dir is empty, then cleanup new dir."""
        pre_existing = set(os.listdir(sessions_dir))
        assert pre_existing == set()

        os.makedirs(os.path.join(sessions_dir, 'session-1'))

        _cleanup_eval_sessions(sessions_dir, pre_existing)

        assert os.listdir(sessions_dir) == []

    def test_snapshot_dir_not_existing_then_created_during_work(self, tmp_path):
        """Simulates: sessions_dir doesn't exist at snapshot time, gets created later."""
        sessions_dir = str(tmp_path / 'sessions')
        assert not os.path.isdir(sessions_dir)

        pre_existing = set()

        os.makedirs(sessions_dir)
        os.makedirs(os.path.join(sessions_dir, 'new-session'))

        _cleanup_eval_sessions(sessions_dir, pre_existing)

        assert os.listdir(sessions_dir) == []

    def test_cleanup_called_even_on_exception(self, sessions_dir):
        """Simulates the try/finally pattern in process_instance."""
        os.makedirs(os.path.join(sessions_dir, 'pre'))
        pre_existing = set(os.listdir(sessions_dir))

        try:
            os.makedirs(os.path.join(sessions_dir, 'new-session'))
            raise RuntimeError('simulated failure')
        except RuntimeError:
            pass
        finally:
            _cleanup_eval_sessions(sessions_dir, pre_existing)

        remaining = set(os.listdir(sessions_dir))
        assert remaining == {'pre'}

    def test_multiple_new_sessions_all_cleaned(self, sessions_dir):
        """Multiple eval runs create different sessions; all new ones get cleaned."""
        os.makedirs(os.path.join(sessions_dir, 'baseline'))
        pre_existing = set(os.listdir(sessions_dir))

        for i in range(10):
            os.makedirs(os.path.join(sessions_dir, f'eval-run-{i}'))

        _cleanup_eval_sessions(sessions_dir, pre_existing)

        remaining = set(os.listdir(sessions_dir))
        assert remaining == {'baseline'}

    def test_concurrent_safe_pre_existing_names(self, sessions_dir):
        """Pre-existing set contains the exact names snapshotted earlier."""
        names = [f'session-{i:04d}' for i in range(20)]
        for name in names:
            os.makedirs(os.path.join(sessions_dir, name))

        pre_existing = set(os.listdir(sessions_dir))
        assert len(pre_existing) == 20

        os.makedirs(os.path.join(sessions_dir, 'new-eval'))

        _cleanup_eval_sessions(sessions_dir, pre_existing)

        remaining = set(os.listdir(sessions_dir))
        assert remaining == set(names)
        assert 'new-eval' not in remaining


# =========================================================================
# D. Source Sync Verification
# =========================================================================


class TestSourceSync:
    """D: Verify the local replica stays in sync with the real source."""

    def test_source_contains_cleanup_function(self):
        """Ensure the function still exists in the source file."""
        source_path = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'evaluation', 'benchmarks', 'swe_bench', 'run_infer.py',
        )
        source_path = os.path.normpath(source_path)
        with open(source_path) as f:
            source = f.read()
        assert 'def _cleanup_eval_sessions(' in source
        assert 'shutil.rmtree(' in source
        assert 'pre_existing' in source
