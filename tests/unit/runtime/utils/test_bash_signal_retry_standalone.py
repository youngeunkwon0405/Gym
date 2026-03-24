#!/usr/bin/env python3
"""Standalone script to test C-c signal retry behavior in BashSession.

Run directly:
    python tests/unit/runtime/utils/test_bash_signal_retry_standalone.py

No pytest needed. Prints colored pass/fail for each test case.
"""

import os
import sys
import tempfile
import time
import traceback

# Add the repo root to sys.path so we can import openhands
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

from openhands.events.action import CmdRunAction
from openhands.runtime.utils.bash import BashCommandStatus, BashSession

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
results: list[tuple[str, bool, str]] = []


def run_test(name, fn):
    print(f"\n{CYAN}{BOLD}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}  TEST: {name}{RESET}")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}")
    try:
        fn()
        results.append((name, True, ""))
        print(f"\n  {GREEN}{BOLD}PASSED{RESET}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"\n  {RED}{BOLD}FAILED: {e}{RESET}")
        traceback.print_exc()


def write_script(content, tmp_dir, name="script.py"):
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Test 1: Basic C-c on a simple loop
# ---------------------------------------------------------------------------
def test_basic_ctrl_c():
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=3)
    session.initialize()
    try:
        print("  Starting: while true; do echo looping; sleep 4; done")
        obs = session.execute(
            CmdRunAction("while true; do echo 'looping'; sleep 4; done")
        )
        print(f"  Got timeout (exit_code={obs.metadata.exit_code})")
        assert obs.metadata.exit_code == -1, f"Expected -1, got {obs.metadata.exit_code}"
        assert "looping" in obs.content

        print("  Sending C-c...")
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        print(f"  Result: exit_code={obs.metadata.exit_code}")
        print(f"  Content: {obs.content!r}")
        assert obs.metadata.exit_code in (1, 130), f"Expected 1 or 130, got {obs.metadata.exit_code}"
        assert "CTRL+C was sent" in obs.metadata.suffix

        print("  Verifying session recovery...")
        obs = session.execute(CmdRunAction("echo recovered"))
        assert "recovered" in obs.content
        print(f"  Session recovered OK (exit_code={obs.metadata.exit_code})")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 2: Process needs 2 SIGINTs (retry kicks in)
# ---------------------------------------------------------------------------
def test_two_sigints_needed():
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    print(f'SIGINT #{count[0]} caught, ignoring', flush=True)
    if count[0] >= 2:
        print('Second SIGINT received, exiting', flush=True)
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    time.sleep(0.1)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            print(f"  Starting: python3 {os.path.basename(script_path)}")
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            print(f"  Got timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c (retry should auto-send second C-c at ~2s)...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Content: {obs.content!r}")

            assert obs.metadata.exit_code in (1, 130), (
                f"Expected 1 or 130, got {obs.metadata.exit_code}"
            )
            assert "SIGINT #1 caught" in obs.content, "Should see first SIGINT handler output"
            assert "SIGINT #2 caught" in obs.content, "Retry should have sent second SIGINT"
            print(f"  Process killed by retry in {elapsed:.1f}s")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 3: Piped command — python ... | tail -20
# ---------------------------------------------------------------------------
def test_piped_command_tail():
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    try:
        print(f'SIGINT #{count[0]}', flush=True)
    except BrokenPipeError:
        pass
    if count[0] >= 2:
        sys.exit(130)
signal.signal(signal.SIGINT, handler)
i = 0
while True:
    i += 1
    try:
        print(f'test output line {i}', flush=True)
    except BrokenPipeError:
        pass
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(script, tmp_dir, "long_running.py")
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | tail -20"
            print(f"  Starting: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            print(f"  Got timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c (tail dies first, pipe breaks, retry kills python)...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Content: {obs.content!r}")

            assert obs.metadata.exit_code in (0, 1, 130, 141), (
                f"Expected process to exit, got {obs.metadata.exit_code}"
            )
            assert "CTRL+C was sent" in obs.metadata.suffix
            print(f"  Piped command killed in {elapsed:.1f}s")

            obs = session.execute(CmdRunAction("echo 'pipe test passed'"))
            assert "pipe test passed" in obs.content
            print("  Session recovered OK")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 4: Piped command — python ... | grep ERROR
# ---------------------------------------------------------------------------
def test_piped_command_grep():
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    if count[0] >= 2:
        sys.exit(1)
signal.signal(signal.SIGINT, handler)
while True:
    try:
        print('running...', flush=True)
    except BrokenPipeError:
        pass
    time.sleep(0.5)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(script, tmp_dir, "grep_test.py")
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | grep ERROR"
            print(f"  Starting: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            print(f"  Got timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Content: {obs.content!r}")

            assert obs.metadata.exit_code in (0, 1, 130, 141)
            assert "CTRL+C was sent" in obs.metadata.suffix
            print(f"  Piped grep command killed in {elapsed:.1f}s")

            obs = session.execute(CmdRunAction("echo 'grep test passed'"))
            assert "grep test passed" in obs.content
            print("  Session recovered OK")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 5: Process that fully ignores SIGINT, then C-z fallback
# ---------------------------------------------------------------------------
def test_sigint_ignored_cz_fallback():
    script = r"""
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
print('SIGINT fully ignored', flush=True)
while True:
    time.sleep(0.1)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            print(f"  Starting: python3 {os.path.basename(script_path)}")
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            print(f"  Got timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c (process ignores SIGINT, all retries will fail)...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  C-c result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            assert obs.metadata.exit_code == -1, "Process should still be running"

            print("  Sending C-z (SIGTSTP) to suspend the process...")
            obs = session.execute(CmdRunAction("C-z", is_input=True))
            print(f"  C-z result: exit_code={obs.metadata.exit_code}")
            print(f"  Content: {obs.content!r}")
            assert session.prev_status == BashCommandStatus.COMPLETED
            assert obs.metadata.exit_code in (0, 20, 148)
            print("  Process suspended successfully")

            obs = session.execute(CmdRunAction("echo 'suspend test passed'"))
            assert "suspend test passed" in obs.content
            print("  Session recovered OK")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 6: C-c without is_input gets blocked
# ---------------------------------------------------------------------------
def test_ctrl_c_without_is_input():
    session = BashSession(work_dir=os.getcwd(), no_change_timeout_seconds=2)
    session.initialize()
    try:
        print("  Starting: sleep 30")
        obs = session.execute(CmdRunAction("sleep 30"))
        assert obs.metadata.exit_code == -1

        print("  Sending C-c WITHOUT is_input=True (should be blocked)...")
        obs = session.execute(CmdRunAction("C-c"))
        print(f"  Result: {obs.metadata.suffix[:80]!r}...")
        assert "is NOT executed" in obs.metadata.suffix
        print("  Correctly blocked!")

        print("  Sending C-c WITH is_input=True (should work)...")
        obs = session.execute(CmdRunAction("C-c", is_input=True))
        print(f"  Result: exit_code={obs.metadata.exit_code}")
        assert obs.metadata.exit_code in (0, 1, 130, 143)
        print("  Process killed")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 7: Pytest-like behavior (graceful shutdown on 1st, abort on 2nd)
# ---------------------------------------------------------------------------
def test_pytest_like_behavior():
    script = r"""
import signal, sys, time
count = [0]
def handler(sig, frame):
    count[0] += 1
    if count[0] == 1:
        for i in range(3):
            print(f'Shutting down... step {i+1}', flush=True)
            time.sleep(0.1)
        print('Waiting for second interrupt to abort...', flush=True)
    else:
        print('Aborted!', flush=True)
        sys.exit(130)
signal.signal(signal.SIGINT, handler)
print('Running tests...', flush=True)
while True:
    time.sleep(0.1)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(script, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            print(f"  Starting: python3 {os.path.basename(script_path)}")
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            assert obs.metadata.exit_code == -1
            assert "Running tests..." in obs.content

            print("  Sending C-c (1st shows shutdown, 2nd aborts)...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Content: {obs.content!r}")

            assert obs.metadata.exit_code == 130
            assert "Shutting down" in obs.content
            assert "Aborted!" in obs.content
            print(f"  Pytest-like process killed in {elapsed:.1f}s")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{BOLD}Signal Retry Test Suite{RESET}")
    print(f"Testing BashSession C-c retry mechanism\n")

    run_test("1. Basic C-c on simple loop", test_basic_ctrl_c)
    run_test("2. Process needs 2 SIGINTs (retry at 2s)", test_two_sigints_needed)
    run_test("3. Piped command: python ... | tail -20", test_piped_command_tail)
    run_test("4. Piped command: python ... | grep ERROR", test_piped_command_grep)
    run_test("5. SIGINT ignored, C-z fallback", test_sigint_ignored_cz_fallback)
    run_test("6. C-c without is_input blocked", test_ctrl_c_without_is_input)
    run_test("7. Pytest-like (shutdown + abort)", test_pytest_like_behavior)

    # Summary
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, err in results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}")
        if err:
            print(f"         {RED}{err}{RESET}")

    print(f"\n  {BOLD}{passed} passed, {failed} failed{RESET}")
    if failed:
        print(f"  {RED}Some tests failed!{RESET}")
        sys.exit(1)
    else:
        print(f"  {GREEN}All tests passed!{RESET}")
        sys.exit(0)
