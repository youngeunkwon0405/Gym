#!/usr/bin/env python3
"""Test C-c behavior with Django-like test runners and pipes.

Simulates the exact scenario from SWE-bench evaluation:
  python tests/runtests.py --tag index 2>&1 | tail -20

Run directly:
    python tests/unit/runtime/utils/test_bash_signal_django.py

Or with pytest:
    python -m pytest tests/unit/runtime/utils/test_bash_signal_django.py -v -s
"""

import os
import sys
import tempfile
import time
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

from openhands.events.action import CmdRunAction
from openhands.runtime.utils.bash import BashCommandStatus, BashSession

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

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


def write_script(content, tmp_dir, name="runtests.py"):
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Django runtests.py simulator
#
# Django's runtests.py uses unittest which catches KeyboardInterrupt:
#   - 1st Ctrl+C: unittest catches it, prints partial results, keeps going
#   - 2nd Ctrl+C: raises KeyboardInterrupt again, exits
#
# This is the actual behavior from cpython's unittest/runner.py:
#   installHandler() registers a handler that:
#     - On 1st SIGINT: sets a flag, lets current test finish
#     - On 2nd SIGINT: raises KeyboardInterrupt immediately
# ---------------------------------------------------------------------------

DJANGO_RUNTESTS_SIMULATOR = r"""
import signal
import sys
import time

sigint_count = [0]
original_handler = signal.getsignal(signal.SIGINT)

def unittest_sigint_handler(sig, frame):
    '''Simulates unittest.installHandler() behavior'''
    sigint_count[0] += 1
    if sigint_count[0] == 1:
        # 1st Ctrl+C: unittest catches it, prints partial result, continues
        print('\n\nInterrupted by user (1st)', flush=True)
        print('Ran 42 tests in 12.345s\n', flush=True)
        print('INTERRUPTED', flush=True)
        # Re-install the original handler so 2nd Ctrl+C raises KeyboardInterrupt
        signal.signal(signal.SIGINT, original_handler)
    # 2nd Ctrl+C will hit the default handler and raise KeyboardInterrupt

signal.signal(signal.SIGINT, unittest_sigint_handler)

# Simulate running tests
print('Running Django tests...', flush=True)
print('Creating test database...', flush=True)
print('System check identified no issues.', flush=True)

i = 0
try:
    while True:
        i += 1
        print(f'test_something_{i} ... ok', flush=True)
        time.sleep(0.3)
except KeyboardInterrupt:
    print('\n\nKeyboardInterrupt caught by runtests.py', flush=True)
    print('Destroying test database...', flush=True)
    time.sleep(0.2)
    print('Done.', flush=True)
    sys.exit(1)
"""

DJANGO_RUNTESTS_CATCHES_BROKENPIPE = r"""
import signal
import sys
import time
import errno

sigint_count = [0]

def handler(sig, frame):
    sigint_count[0] += 1
    if sigint_count[0] >= 2:
        sys.exit(1)
    # 1st Ctrl+C: try to print, may hit BrokenPipeError
    try:
        print('Interrupted', flush=True)
    except (BrokenPipeError, IOError):
        pass

signal.signal(signal.SIGINT, handler)

print('Running Django tests...', flush=True)
i = 0
try:
    while True:
        i += 1
        try:
            print(f'test_{i} ... ok', flush=True)
        except (BrokenPipeError, IOError):
            # Pipe broke (tail died), keep running silently
            time.sleep(0.3)
            continue
        time.sleep(0.3)
except KeyboardInterrupt:
    sys.exit(1)
"""


# ---------------------------------------------------------------------------
# Test 1: Django runtests WITHOUT pipe — C-c should work on 2nd signal
# ---------------------------------------------------------------------------
def test_django_runtests_no_pipe():
    """python tests/runtests.py (no pipe)

    Django's unittest handler catches 1st SIGINT, re-installs default handler.
    2nd SIGINT raises KeyboardInterrupt, runtests.py exits gracefully.
    The retry mechanism sends the 2nd C-c automatically.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(DJANGO_RUNTESTS_SIMULATOR, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            print(f"  Command: python3 {os.path.basename(script_path)}")
            obs = session.execute(CmdRunAction(f"python3 {script_path}"))
            print(f"  Timeout (exit_code={obs.metadata.exit_code})")
            print(f"  Output: {obs.content[:100]!r}...")
            assert obs.metadata.exit_code == -1
            assert "Running Django tests" in obs.content

            print("  Sending C-c...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Output: {obs.content!r}")

            assert obs.metadata.exit_code in (1, 130), (
                f"Expected 1 or 130, got {obs.metadata.exit_code}. "
                f"Content: {obs.content!r}"
            )
            assert session.prev_status == BashCommandStatus.COMPLETED

            # Verify we see Django's shutdown sequence
            assert "Interrupted by user" in obs.content or "KeyboardInterrupt" in obs.content
            print(f"  Django tests interrupted in {elapsed:.1f}s")

            obs = session.execute(CmdRunAction("echo 'session ok'"))
            assert "session ok" in obs.content
            print("  Session recovered")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 2: Django runtests WITH | tail -20 — the exact SWE-bench pattern
# ---------------------------------------------------------------------------
def test_django_runtests_piped_tail():
    """python tests/runtests.py --tag index 2>&1 | tail -20

    This is the exact command pattern from the screenshot.
    1st C-c: tail dies (SIGINT), pipe breaks, python catches SIGINT but
             can't write output through broken pipe → pane shows NO change.
    Retry 2nd C-c: python gets 2nd SIGINT, exits → shell shows PS1.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(DJANGO_RUNTESTS_CATCHES_BROKENPIPE, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | tail -20"
            print(f"  Command: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            print(f"  Timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c (tail dies, pipe breaks, retry kills python)...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Output: {obs.content!r}")

            assert obs.metadata.exit_code in (0, 1, 130, 141), (
                f"Expected process exit, got {obs.metadata.exit_code}. "
                f"Content: {obs.content!r}"
            )
            assert session.prev_status == BashCommandStatus.COMPLETED
            print(f"  Piped Django command killed in {elapsed:.1f}s")

            obs = session.execute(CmdRunAction("echo 'session ok'"))
            assert "session ok" in obs.content
            print("  Session recovered")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 3: Django runtests WITH | tail -20 — using the realistic simulator
# ---------------------------------------------------------------------------
def test_django_runtests_realistic_piped():
    """Same as test 2 but with the more realistic unittest handler.

    The unittest handler re-installs the default SIGINT handler after 1st Ctrl+C.
    So the 2nd C-c from retry raises KeyboardInterrupt directly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(DJANGO_RUNTESTS_SIMULATOR, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | tail -20"
            print(f"  Command: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            print(f"  Timeout (exit_code={obs.metadata.exit_code})")
            assert obs.metadata.exit_code == -1

            print("  Sending C-c...")
            start = time.time()
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            elapsed = time.time() - start
            print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")
            print(f"  Output: {obs.content!r}")

            assert obs.metadata.exit_code in (0, 1, 130, 141), (
                f"Expected process exit, got {obs.metadata.exit_code}"
            )
            assert session.prev_status == BashCommandStatus.COMPLETED
            print(f"  Realistic piped Django command killed in {elapsed:.1f}s")

            obs = session.execute(CmdRunAction("echo 'session ok'"))
            assert "session ok" in obs.content
            print("  Session recovered")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 4: Django runtests WITH | head -50
# ---------------------------------------------------------------------------
def test_django_runtests_piped_head():
    """python tests/runtests.py 2>&1 | head -50

    head exits after reading 50 lines. If the test produces > 50 lines,
    head closes the pipe. python gets SIGPIPE but may keep running.
    Then C-c needs to kill python.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(DJANGO_RUNTESTS_CATCHES_BROKENPIPE, tmp_dir)
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=5)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | head -5"
            print(f"  Command: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            print(f"  exit_code={obs.metadata.exit_code}")
            print(f"  Output: {obs.content[:200]!r}")

            if obs.metadata.exit_code == -1:
                # head closed pipe, python still running
                print("  head exited, python still running — sending C-c...")
                start = time.time()
                obs = session.execute(CmdRunAction("C-c", is_input=True))
                elapsed = time.time() - start
                print(f"  Result after {elapsed:.1f}s: exit_code={obs.metadata.exit_code}")

                assert obs.metadata.exit_code in (0, 1, 130, 141)
                assert session.prev_status == BashCommandStatus.COMPLETED
                print(f"  head-piped command killed in {elapsed:.1f}s")
            else:
                print("  Command completed on its own (head got enough lines)")

            obs = session.execute(CmdRunAction("echo 'session ok'"))
            assert "session ok" in obs.content
            print("  Session recovered")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test 5: Verify OLD behavior would fail (single C-c, short timeout)
#
# This simulates what the agent experienced: a single C-c with no retry.
# We use no_change_timeout=2 which is shorter than SIGNAL_RETRY_INTERVAL=2,
# so the retry never fires — demonstrating the old broken behavior.
# ---------------------------------------------------------------------------
def test_old_behavior_would_fail():
    """Demonstrate that a single C-c (no retry) fails on piped Django commands.

    Uses no_change_timeout=1 (< SIGNAL_RETRY_INTERVAL=2) so the retry
    never fires. This is equivalent to the old behavior.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = write_script(DJANGO_RUNTESTS_CATCHES_BROKENPIPE, tmp_dir)
        # Timeout shorter than retry interval → retry never fires
        session = BashSession(work_dir=tmp_dir, no_change_timeout_seconds=1)
        session.initialize()
        try:
            cmd = f"python3 {script_path} 2>&1 | tail -20"
            print(f"  Command: {cmd}")
            obs = session.execute(CmdRunAction(cmd))
            assert obs.metadata.exit_code == -1

            print("  Sending C-c (no retry will fire — timeout too short)...")
            obs = session.execute(CmdRunAction("C-c", is_input=True))
            print(f"  Result: exit_code={obs.metadata.exit_code}")
            print(f"  Output: {obs.content!r}")

            # This demonstrates the OLD behavior: C-c times out, process still running
            assert obs.metadata.exit_code == -1, (
                "Expected -1 (process still running) to demonstrate old broken behavior"
            )
            assert session.prev_status == BashCommandStatus.NO_CHANGE_TIMEOUT
            print("  CONFIRMED: single C-c fails on piped command (old behavior)")
            print("  Process is still running — this is what the agent saw!")

            # Clean up: use C-z to suspend
            obs = session.execute(CmdRunAction("C-z", is_input=True))
            print(f"  C-z cleanup: exit_code={obs.metadata.exit_code}")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{BOLD}Django C-c Signal Test Suite{RESET}")
    print(f"Simulates: python tests/runtests.py --tag index 2>&1 | tail -20\n")

    run_test("1. Django runtests — no pipe", test_django_runtests_no_pipe)
    run_test("2. Django runtests — piped through tail -20", test_django_runtests_piped_tail)
    run_test("3. Django runtests — realistic unittest handler + tail", test_django_runtests_realistic_piped)
    run_test("4. Django runtests — piped through head -5", test_django_runtests_piped_head)
    run_test("5. OLD behavior demo — single C-c fails on pipe", test_old_behavior_would_fail)

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
        sys.exit(1)
    else:
        print(f"  {GREEN}All tests passed!{RESET}")
        sys.exit(0)
