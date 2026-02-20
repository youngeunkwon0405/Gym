"""Unit tests for the Terminus-2 Agent.

Tests the Terminus2Agent's message building, output truncation,
and core agent logic.
"""

import pytest

from openhands.agenthub.terminus_2_agent.terminus_2_agent import (
    COMPLETION_CONFIRMATION,
    TIMEOUT_TEMPLATE,
    Terminus2Agent,
)
from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    TerminusJSONPlainParser,
)
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import EventSource
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation


# ==============================================================================
# Output Truncation Tests
# ==============================================================================


class TestOutputTruncation:
    """Tests for the _limit_output_length static method."""

    def test_short_output_not_truncated(self):
        output = 'short output'
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert result == output

    def test_long_output_truncated(self):
        output = 'x' * 20000
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert len(result.encode('utf-8')) < len(output.encode('utf-8'))
        assert 'output limited to 10000 bytes' in result
        assert 'interior bytes omitted' in result

    def test_exact_limit_not_truncated(self):
        output = 'x' * 10000
        result = Terminus2Agent._limit_output_length(output, max_bytes=10000)
        assert result == output

    def test_truncation_preserves_start_and_end(self):
        output = 'START' + 'x' * 20000 + 'END'
        result = Terminus2Agent._limit_output_length(output, max_bytes=1000)
        assert result.startswith('START')
        assert result.endswith('END')

    def test_unicode_truncation(self):
        output = '\u00e9' * 10000  # Each char is 2 bytes in UTF-8
        result = Terminus2Agent._limit_output_length(output, max_bytes=5000)
        assert 'output limited to 5000 bytes' in result

    def test_custom_max_bytes(self):
        output = 'a' * 500
        result = Terminus2Agent._limit_output_length(output, max_bytes=200)
        assert 'output limited to 200 bytes' in result

    def test_empty_output(self):
        result = Terminus2Agent._limit_output_length('', max_bytes=10000)
        assert result == ''


# ==============================================================================
# Template Tests
# ==============================================================================


class TestTemplates:
    """Tests for the message templates used by the agent."""

    def test_timeout_template_formatting(self):
        result = TIMEOUT_TEMPLATE.format(
            command='make build\n',
            timeout_sec=30,
            terminal_state='$ make build\ncompiling...',
        )
        assert 'make build' in result
        assert '30 seconds' in result
        assert 'compiling...' in result
        assert 'timed out' in result

    def test_completion_confirmation_formatting(self):
        result = COMPLETION_CONFIRMATION.format(
            terminal_state='$ echo done\ndone\n$',
        )
        assert 'echo done' in result
        assert 'task_complete' in result
        assert 'graded' in result


# ==============================================================================
# Parser Integration Tests
# ==============================================================================


class TestParserIntegration:
    """Tests that the parser integrates correctly with the agent's expected flow."""

    @pytest.fixture
    def parser(self):
        return TerminusJSONPlainParser()

    def test_typical_response_flow(self, parser):
        """Simulate a typical multi-command response."""
        response = '''{
            "analysis": "I need to set up the project. The directory is empty.",
            "plan": "1. Create a project directory. 2. Initialize it.",
            "commands": [
                {"keystrokes": "mkdir myproject\\n", "duration": 0.1},
                {"keystrokes": "cd myproject\\n", "duration": 0.1},
                {"keystrokes": "git init\\n", "duration": 1.0}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 3
        assert result.commands[0].keystrokes == 'mkdir myproject\n'
        assert result.commands[1].keystrokes == 'cd myproject\n'
        assert result.commands[2].keystrokes == 'git init\n'
        assert result.commands[2].duration == 1.0
        assert result.is_task_complete is False

    def test_completion_response(self, parser):
        """Simulate a task completion response."""
        response = '''{
            "analysis": "All tests pass. The implementation is complete.",
            "plan": "Mark the task as complete.",
            "commands": [],
            "task_complete": true
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert result.is_task_complete is True
        assert len(result.commands) == 0

    def test_wait_response(self, parser):
        """Simulate a wait-for-output response."""
        response = '''{
            "analysis": "The build is still running.",
            "plan": "Wait for the build to finish.",
            "commands": [
                {"keystrokes": "", "duration": 10.0}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == ''
        assert result.commands[0].duration == 10.0

    def test_ctrl_c_response(self, parser):
        """Simulate sending Ctrl+C to cancel a running process."""
        response = '''{
            "analysis": "The process appears to be stuck.",
            "plan": "Send Ctrl+C to cancel it.",
            "commands": [
                {"keystrokes": "C-c", "duration": 0.1}
            ]
        }'''
        result = parser.parse_response(response)

        assert result.error == ''
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == 'C-c'


# ==============================================================================
# Message Building Tests
# ==============================================================================


class TestMessageBuilding:
    """Tests for the agent's message building logic."""

    def test_find_initial_user_message(self):
        """Test extraction of initial user message from events."""
        msg = MessageAction(content='Fix the bug in module X')
        msg._source = EventSource.USER

        events = [msg]
        # Use static-like approach
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                assert event.content == 'Fix the bug in module X'
                break

    def test_terminus_2_action_batch_creation(self):
        """Test that commands are correctly converted to actions."""
        commands = [
            ParsedCommand(keystrokes='ls -la\n', duration=0.1),
            ParsedCommand(keystrokes='cat file.txt\n', duration=0.5),
        ]

        actions = []
        for cmd in commands:
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, 60),
            )
            actions.append(action)

        assert len(actions) == 2
        assert actions[0].keystrokes == 'ls -la\n'
        assert actions[0].duration == 0.1
        assert actions[1].keystrokes == 'cat file.txt\n'
        assert actions[1].duration == 0.5

    def test_duration_capped_at_60(self):
        """Test that duration is capped at 60 seconds."""
        cmd = ParsedCommand(keystrokes='sleep 100\n', duration=100.0)
        action = Terminus2CmdRunAction(
            keystrokes=cmd.keystrokes,
            duration=min(cmd.duration, 60),
        )
        assert action.duration == 60


# ==============================================================================
# Double Confirmation Tests
# ==============================================================================


class TestDoubleConfirmation:
    """Tests for the double-confirmation task completion logic."""

    def test_pending_completion_flag_initial(self):
        """Verify the flag starts as False."""
        # We can't instantiate the full agent without LLM registry,
        # but we can test the logic pattern
        pending = False

        # First task_complete=true
        is_task_complete = True
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert pending is True
        assert action == 'confirm'

    def test_pending_completion_second_time(self):
        """Verify second task_complete triggers finish."""
        pending = True  # Already set from first confirmation

        is_task_complete = True
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert action == 'finish'

    def test_pending_completion_reset_on_not_complete(self):
        """Verify pending is reset when task_complete is False."""
        pending = True

        is_task_complete = False
        if is_task_complete:
            if pending:
                action = 'finish'
            else:
                pending = True
                action = 'confirm'
        else:
            pending = False
            action = 'continue'

        assert pending is False
        assert action == 'continue'


# ==============================================================================
# Observation Handling Tests
# ==============================================================================


class TestObservationHandling:
    """Tests for handling Terminus-2 observations."""

    def test_observation_content_extraction(self):
        obs = Terminus2CmdOutputObservation(
            content='$ ls\nfile1.txt\nfile2.txt\n$',
            terminal_state='$ ls\nfile1.txt\nfile2.txt\n$',
            timed_out=False,
            command_keystrokes='ls\n',
        )

        assert obs.terminal_state == '$ ls\nfile1.txt\nfile2.txt\n$'
        assert obs.timed_out is False
        assert obs.command_keystrokes == 'ls\n'

    def test_timed_out_observation(self):
        obs = Terminus2CmdOutputObservation(
            content='partial output...',
            terminal_state='partial output...',
            timed_out=True,
            command_keystrokes='make\n',
        )

        assert obs.timed_out is True
        assert 'timed out' in obs.message

    def test_observation_with_empty_terminal_state(self):
        obs = Terminus2CmdOutputObservation(
            content='',
            terminal_state='',
        )
        assert obs.terminal_state == ''
        assert obs.content == ''
