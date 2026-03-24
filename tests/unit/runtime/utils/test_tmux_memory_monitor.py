"""Tests for TmuxMemoryMonitor.kill_inner_processes — verifying it kills
commands but keeps the shell alive so the pane remains usable.

Run:
    python -m pytest tests/unit/runtime/utils/test_tmux_memory_monitor.py -v -s
"""

import os
import tempfile
import time

from openhands.events.action import CmdRunAction
from openhands.runtime.utils.bash import BashCommandStatus, BashSession


def _write_script(content: str, tmp_dir: str, name: str = "script.py") -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Test 1: kill_inner_processes kills the command, shell survives
# ---------------------------------------------------------------------------


def test_memory_monitor_kills_command_shell_survives():
    """After kill_inner_processes, the shell is alive and the session is usable."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        # Start a long-running process
        obs = session.execute(CmdRunAction("sleep 300"))
        assert obs.metadata.exit_code == -1
        assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT

        # Simulate what the memory monitor does
        session.memory_monitor.kill_inner_processes()
        time.sleep(1)

        # The pane should still be alive — send empty to check output
        obs = session.execute(CmdRunAction("", is_input=True))
        # The process was killed, shell should show PS1
        # Either it completed (PS1 detected) or we can try a command
        if session.prev_status != BashCommandStatus.COMPLETED:
            # Might need a moment for shell to recover
            time.sleep(1)
            obs = session.execute(CmdRunAction("", is_input=True))

        # Now run a normal command — this is the key test
        obs = session.execute(CmdRunAction("echo 'shell survived'"))
        assert "shell survived" in obs.content
        assert obs.metadata.exit_code == 0
        assert session.prev_status == BashCommandStatus.COMPLETED
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 2: kill_inner_processes on a piped command
# ---------------------------------------------------------------------------


def test_memory_monitor_kills_piped_command():
    """Piped command (python | tail) is killed, shell survives."""
    script = r"""
import time
while True:
    print('output', flush=True)
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=3)
        session.initialize()
        try:
            obs = session.execute(
                CmdRunAction(f"python3 {script_path} 2>&1 | tail -20")
            )
            assert obs.metadata.exit_code == -1

            # Memory monitor kills
            session.memory_monitor.kill_inner_processes()
            time.sleep(1)

            # Recover
            obs = session.execute(CmdRunAction("", is_input=True))
            if session.prev_status != BashCommandStatus.COMPLETED:
                time.sleep(1)
                obs = session.execute(CmdRunAction("", is_input=True))

            obs = session.execute(CmdRunAction("echo 'pipe kill ok'"))
            assert "pipe kill ok" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 3: kill_inner_processes on SIGINT-immune process
# ---------------------------------------------------------------------------


def test_memory_monitor_kills_sigint_immune_process():
    """Process that ignores SIGINT is killed by memory monitor, shell survives."""
    script = r"""
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
while True:
    time.sleep(0.1)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=3)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            assert obs.metadata.exit_code == -1

            session.memory_monitor.kill_inner_processes()
            time.sleep(1)

            obs = session.execute(CmdRunAction("", is_input=True))
            if session.prev_status != BashCommandStatus.COMPLETED:
                time.sleep(1)
                obs = session.execute(CmdRunAction("", is_input=True))

            obs = session.execute(CmdRunAction("echo 'immune kill ok'"))
            assert "immune kill ok" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 4: kill_inner_processes then C-c escalation works after
# ---------------------------------------------------------------------------


def test_memory_monitor_then_ctrl_c_still_works():
    """After memory monitor kills a process, C-c still works on subsequent commands."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        # First: memory monitor kills a process
        obs = session.execute(CmdRunAction("sleep 300"))
        assert obs.metadata.exit_code == -1

        session.memory_monitor.kill_inner_processes()
        time.sleep(1)

        obs = session.execute(CmdRunAction("", is_input=True))
        if session.prev_status != BashCommandStatus.COMPLETED:
            time.sleep(1)
            obs = session.execute(CmdRunAction("", is_input=True))

        # Verify recovery
        obs = session.execute(CmdRunAction("echo checkpoint1"))
        assert "checkpoint1" in obs.content

        # Second: start another long process, use C-c
        obs = session.execute(CmdRunAction("sleep 300"))
        assert obs.metadata.exit_code == -1

        obs = session.execute(CmdRunAction("C-c", is_input=True))
        assert obs.metadata.exit_code in (0, 1, 130, 137)
        assert session.prev_status == BashCommandStatus.COMPLETED

        obs = session.execute(CmdRunAction("echo checkpoint2"))
        assert "checkpoint2" in obs.content
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 5: kill_inner_processes when no command is running (idle shell)
# ---------------------------------------------------------------------------


def test_memory_monitor_idle_shell():
    """kill_inner_processes on idle shell doesn't break anything."""
    session = BashSession(work_dir=os.getcwd())
    session.initialize()
    try:
        # Run a command first so the shell is in a known state
        obs = session.execute(CmdRunAction("echo hello"))
        assert "hello" in obs.content

        # Kill with no running command — shell has no children
        session.memory_monitor.kill_inner_processes()
        time.sleep(0.5)

        # Shell should still work
        obs = session.execute(CmdRunAction("echo 'still alive'"))
        assert "still alive" in obs.content
        assert obs.metadata.exit_code == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 6: kill_inner_processes on process with child processes
# ---------------------------------------------------------------------------


def test_memory_monitor_kills_process_tree():
    """Process that spawns children — all get killed, shell survives."""
    script = r"""
import subprocess, time
procs = []
for i in range(3):
    p = subprocess.Popen(['sleep', '300'])
    procs.append(p)
# Parent also sleeps
time.sleep(300)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = _write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=3)
        session.initialize()
        try:
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            assert obs.metadata.exit_code == -1

            # Give child processes time to spawn
            time.sleep(1)

            session.memory_monitor.kill_inner_processes()
            time.sleep(1)

            obs = session.execute(CmdRunAction("", is_input=True))
            if session.prev_status != BashCommandStatus.COMPLETED:
                time.sleep(1)
                obs = session.execute(CmdRunAction("", is_input=True))

            obs = session.execute(CmdRunAction("echo 'tree kill ok'"))
            assert "tree kill ok" in obs.content
            assert obs.metadata.exit_code == 0
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 7: multiple kill_inner_processes calls don't break the pane
# ---------------------------------------------------------------------------


def test_memory_monitor_multiple_kills():
    """Multiple kill_inner_processes calls in sequence — pane stays alive."""
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        for i in range(3):
            obs = session.execute(CmdRunAction("sleep 300"))
            assert obs.metadata.exit_code == -1

            session.memory_monitor.kill_inner_processes()
            time.sleep(1)

            obs = session.execute(CmdRunAction("", is_input=True))
            if session.prev_status != BashCommandStatus.COMPLETED:
                time.sleep(1)
                obs = session.execute(CmdRunAction("", is_input=True))

            obs = session.execute(CmdRunAction(f"echo 'round {i}'"))
            assert f"round {i}" in obs.content
            assert obs.metadata.exit_code == 0
    finally:
        session.close()
