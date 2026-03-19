"""Tests for tool error conversation flow.

Verifies that FunctionCallNotExistsError (wrong tool name) produces the correct
conversation structure: ASSISTANT(tool_calls) -> USER(error), not ASSISTANT(error).
"""

import json
from unittest.mock import MagicMock

import pytest
from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import (
    response_to_actions as codeact_response_to_actions,
)
from openhands.agenthub.codex_agent.function_calling import (
    response_to_actions as codex_response_to_actions,
)
from openhands.agenthub.loc_agent.function_calling import (
    response_to_actions as loc_response_to_actions,
)
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.events.action import (
    CmdRunAction,
    FunctionCallNotExistsAction,
    MessageAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation import CmdOutputObservation
from openhands.events.observation.commands import CmdOutputMetadata
from openhands.events.tool import ToolCallMetadata
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_config():
    return AgentConfig(
        enable_prompt_extensions=True,
        enable_som_visual_browsing=True,
        disabled_microagents=['disabled_agent'],
    )


@pytest.fixture
def conversation_memory(agent_config):
    prompt_manager = MagicMock(spec=PromptManager)
    prompt_manager.get_system_message.return_value = 'System message'
    prompt_manager.build_workspace_context.return_value = (
        'Formatted repository and runtime info'
    )
    prompt_manager.build_microagent_info.side_effect = lambda triggered_agents: (
        '\n'.join(agent.content for agent in triggered_agents)
        if triggered_agents
        else ''
    )
    return ConversationMemory(agent_config, prompt_manager)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_response(
    tool_calls: list[dict],
    content: str | None = None,
    response_id: str = 'resp-1',
) -> ModelResponse:
    """Build a ModelResponse with the given tool_calls."""
    tc_objs = []
    for tc in tool_calls:
        tc_objs.append(
            {
                'id': tc['id'],
                'type': 'function',
                'function': {
                    'name': tc['name'],
                    'arguments': json.dumps(tc.get('arguments', {})),
                },
            }
        )
    return ModelResponse(
        id=response_id,
        choices=[
            {
                'message': {
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tc_objs,
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def _make_tool_call_metadata(
    tool_call_id: str,
    function_name: str,
    response_id: str = 'resp-1',
    total_calls: int = 1,
    extra_tool_calls: list[dict] | None = None,
) -> ToolCallMetadata:
    """Build ToolCallMetadata with a properly structured model_response."""
    tc_list = [
        {
            'id': tool_call_id,
            'type': 'function',
            'function': {'name': function_name, 'arguments': '{}'},
        }
    ]
    if extra_tool_calls:
        tc_list.extend(extra_tool_calls)
    mock_response = {
        'id': response_id,
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': tc_list,
                }
            }
        ],
        'created': 0,
        'model': 'mock_model',
        'object': 'chat.completion',
        'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
    }
    return ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name=function_name,
        model_response=mock_response,
        total_calls_in_response=total_calls,
    )


def _system_and_user_events():
    """Return (system_message, user_message, initial_user_action) for process_events."""
    sys_msg = SystemMessageAction(content='System message')
    sys_msg._source = EventSource.AGENT
    user_msg = MessageAction(content='Fix this bug')
    user_msg._source = EventSource.USER
    return sys_msg, user_msg, user_msg


# =========================================================================
# A. conversation_memory level – correct role assignment
# =========================================================================


class TestSingleBadToolCall:
    """A1: LLM calls non-existent tool 'bash' — error must be role='tool'."""

    def test_error_appears_as_tool_message(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()

        error_action = MessageAction(
            content='Tool bash is not registered. Please check the tool name.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = _make_tool_call_metadata(
            tool_call_id='tc-1',
            function_name='bash',
            response_id='resp-1',
        )

        events = [sys_msg, user_msg, error_action]
        messages = conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )

        roles = [m.role for m in messages]
        # system, user, assistant(tool_calls), tool(error)
        assert 'tool' in roles, f'Expected a tool-role error message, got roles: {roles}'
        error_msgs = [
            m for m in messages if m.role == 'tool' and 'not registered' in
            (m.content[0].text if m.content and hasattr(m.content[0], 'text') else '')
        ]
        assert len(error_msgs) == 1, f'Expected exactly one tool error message, found {len(error_msgs)}'
        assert error_msgs[0].tool_call_id == 'tc-1'
        assert error_msgs[0].name == 'bash'
        # The assistant message should have the tool_call preserved
        assistant_msgs = [m for m in messages if m.role == 'assistant' and m.tool_calls]
        assert len(assistant_msgs) == 1, 'Assistant message with tool_calls should be preserved'
        assert assistant_msgs[0].tool_calls[0].id == 'tc-1'
        # Must NOT appear as assistant text
        assistant_error_msgs = [
            m for m in messages if m.role == 'assistant' and 'not registered' in
            (m.content[0].text if m.content and hasattr(m.content[0], 'text') else '')
        ]
        assert len(assistant_error_msgs) == 0, 'Error message must not appear as assistant role'


class TestSingleBadToolCallWithThought:
    """A1b: LLM calls non-existent tool but has a thought — thought and tool_calls preserved."""

    def test_assistant_thought_and_tool_calls_preserved(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()

        # Build a tool_call_metadata whose model_response has assistant content
        response_id = 'resp-thought'
        tc_list = [
            {
                'id': 'tc-1',
                'type': 'function',
                'function': {'name': 'bash', 'arguments': '{}'},
            }
        ]
        mock_response = {
            'id': response_id,
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'Let me check this file.',
                        'tool_calls': tc_list,
                    }
                }
            ],
            'created': 0,
            'model': 'mock_model',
            'object': 'chat.completion',
            'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
        }

        error_action = MessageAction(
            content='Tool bash is not registered.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = ToolCallMetadata(
            tool_call_id='tc-1',
            function_name='bash',
            model_response=mock_response,
            total_calls_in_response=1,
        )

        events = [sys_msg, user_msg, error_action]
        messages = conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )

        # The assistant message should have both thought AND tool_calls preserved
        assistant_msgs = [m for m in messages if m.role == 'assistant']
        thought_msgs = [
            m for m in assistant_msgs
            if m.content and hasattr(m.content[0], 'text')
            and 'Let me check this file' in m.content[0].text
        ]
        assert len(thought_msgs) == 1, (
            f'Assistant thought should be preserved. Assistant messages: '
            f'{[(m.content[0].text if m.content else "") for m in assistant_msgs]}'
        )
        # tool_calls are now preserved since the error is role='tool'
        assert thought_msgs[0].tool_calls is not None
        assert thought_msgs[0].tool_calls[0].id == 'tc-1'

        # The error should be a tool response
        tool_msgs = [m for m in messages if m.role == 'tool' and 'not registered' in
                     (m.content[0].text if m.content and hasattr(m.content[0], 'text') else '')]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == 'tc-1'


class TestMultiToolWithOneBadCall:
    """A2: LLM calls 2 tools, one valid and one invalid."""

    def test_valid_tool_call_preserved_invalid_becomes_user_error(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()
        response_id = 'resp-multi'

        shared_tc_list = [
            {
                'id': 'tc-valid',
                'type': 'function',
                'function': {'name': 'execute_bash', 'arguments': '{"command": "ls"}'},
            },
            {
                'id': 'tc-invalid',
                'type': 'function',
                'function': {'name': 'bash', 'arguments': '{}'},
            },
        ]
        mock_response_dict = {
            'id': response_id,
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': shared_tc_list,
                    }
                }
            ],
            'created': 0,
            'model': 'mock_model',
            'object': 'chat.completion',
            'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
        }

        # Valid action (CmdRunAction for execute_bash)
        valid_action = CmdRunAction(command='ls')
        valid_action._source = EventSource.AGENT
        valid_action.tool_call_metadata = ToolCallMetadata(
            tool_call_id='tc-valid',
            function_name='execute_bash',
            model_response=mock_response_dict,
            total_calls_in_response=2,
        )

        # Invalid tool error action
        error_action = MessageAction(
            content='Tool bash is not registered.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = ToolCallMetadata(
            tool_call_id='tc-invalid',
            function_name='bash',
            model_response=mock_response_dict,
            total_calls_in_response=2,
        )

        # Observation for the valid action
        valid_obs = CmdOutputObservation(
            command='ls',
            content='file1.py\nfile2.py',
            command_id=1,
            exit_code=0,
        )
        valid_obs.tool_call_metadata = ToolCallMetadata(
            tool_call_id='tc-valid',
            function_name='execute_bash',
            model_response=mock_response_dict,
            total_calls_in_response=2,
        )

        events = [sys_msg, user_msg, valid_action, error_action, valid_obs]
        messages = conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )

        roles = [m.role for m in messages]

        # Should have: system, user, assistant (with valid tool_call only), tool (valid result), user (error)
        assert 'assistant' in roles
        assistant_msgs = [m for m in messages if m.role == 'assistant']
        # The assistant message should have the valid tool_call but not the invalid one
        for am in assistant_msgs:
            if am.tool_calls:
                tc_names = [tc.function.name for tc in am.tool_calls]
                assert 'execute_bash' in tc_names
                assert 'bash' not in tc_names

        # Error should be a user message
        error_msgs = [
            m for m in messages if m.role == 'user' and 'not registered' in
            (m.content[0].text if m.content and hasattr(m.content[0], 'text') else '')
        ]
        assert len(error_msgs) == 1

        # Valid tool result should be a tool message
        tool_msgs = [m for m in messages if m.role == 'tool']
        assert len(tool_msgs) == 1


class TestMessageActionWithToolMetadataIsToolError:
    """A3: _process_action treats MessageAction(wait_for_response=False, tool_call_metadata) as error."""

    def test_direct_process_action(self, conversation_memory):
        pending: dict[str, Message] = {}
        tool_responses: dict[str, Message] = {}

        error_action = MessageAction(
            content='Tool foo is not registered.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = _make_tool_call_metadata(
            tool_call_id='tc-foo',
            function_name='foo',
        )

        result = conversation_memory._process_action(
            action=error_action,
            pending_tool_call_action_messages=pending,
            tool_call_id_to_message=tool_responses,
            vision_is_active=False,
        )

        assert result == [], '_process_action should return empty list for tool error'
        assert len(pending) == 1, 'Should store assistant message in pending'
        assert 'tc-foo' in tool_responses, 'Should store error in tool_call_id_to_message'
        assert tool_responses['tc-foo'].role == 'user'
        assert 'not registered' in tool_responses['tc-foo'].content[0].text


class TestRegularAgentMessageActionUnchanged:
    """A4: Regular MessageAction from agent (no tool error) still gets role='assistant'."""

    def test_regular_message_action_is_assistant(self, conversation_memory):
        pending: dict[str, Message] = {}
        tool_responses: dict[str, Message] = {}

        regular_action = MessageAction(
            content='I will help you with that.',
            wait_for_response=True,
        )
        regular_action._source = EventSource.AGENT

        result = conversation_memory._process_action(
            action=regular_action,
            pending_tool_call_action_messages=pending,
            tool_call_id_to_message=tool_responses,
            vision_is_active=False,
        )

        assert len(result) == 1
        assert result[0].role == 'assistant'
        assert 'I will help you' in result[0].content[0].text


class TestUserMessageActionUnchanged:
    """A4b: MessageAction from user still gets role='user'."""

    def test_user_message_action_is_user(self, conversation_memory):
        pending: dict[str, Message] = {}
        tool_responses: dict[str, Message] = {}

        user_action = MessageAction(content='Please fix the bug.')
        user_action._source = EventSource.USER

        result = conversation_memory._process_action(
            action=user_action,
            pending_tool_call_action_messages=pending,
            tool_call_id_to_message=tool_responses,
            vision_is_active=False,
        )

        assert len(result) == 1
        assert result[0].role == 'user'


# =========================================================================
# B. function_calling level – correct action type produced
# =========================================================================


class TestCodeactSingleInvalidTool:
    """B1: codeact_agent returns FunctionCallNotExistsAction with correct metadata for invalid tool."""

    def test_returns_function_call_not_exists_action_with_metadata(self):
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'bash', 'arguments': {'command': 'ls'}}],
        )
        actions = codeact_response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, FunctionCallNotExistsAction)
        assert action.function_name == 'bash'
        assert 'not registered' in action.error_message
        assert action.tool_call_metadata is not None
        assert action.tool_call_metadata.tool_call_id == 'tc-1'
        assert action.tool_call_metadata.function_name == 'bash'


class TestCodeactMultiToolWithOneInvalid:
    """B2: codeact_agent multi-tool: valid + invalid."""

    def test_returns_valid_action_and_error_action(self):
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-valid', 'name': 'execute_bash', 'arguments': {'command': 'ls'}},
                {'id': 'tc-invalid', 'name': 'bash', 'arguments': {'command': 'ls'}},
            ],
        )
        actions = codeact_response_to_actions(response)
        assert len(actions) == 2

        valid_action = actions[0]
        assert isinstance(valid_action, CmdRunAction)
        assert valid_action.tool_call_metadata.tool_call_id == 'tc-valid'

        error_action = actions[1]
        assert isinstance(error_action, FunctionCallNotExistsAction)
        assert 'not registered' in error_action.error_message
        assert error_action.tool_call_metadata.tool_call_id == 'tc-invalid'


class TestCodexSingleInvalidTool:
    """B3: codex_agent returns FunctionCallNotExistsAction with correct metadata for invalid tool."""

    def test_returns_function_call_not_exists_action_with_metadata(self):
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {'x': 1}}],
        )
        actions = codex_response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, FunctionCallNotExistsAction)
        assert action.function_name == 'nonexistent'
        assert 'not registered' in action.error_message
        assert action.tool_call_metadata is not None
        assert action.tool_call_metadata.tool_call_id == 'tc-1'


class TestCodexMultiToolWithOneInvalid:
    """B4: codex_agent multi-tool: valid + invalid."""

    def test_returns_valid_action_and_error_action(self):
        from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME

        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-valid', 'name': CODEX_SHELL_COMMAND_TOOL_NAME, 'arguments': {'command': 'ls'}},
                {'id': 'tc-invalid', 'name': 'nonexistent', 'arguments': {}},
            ],
        )
        actions = codex_response_to_actions(response)
        assert len(actions) == 2

        valid_action = actions[0]
        assert isinstance(valid_action, CmdRunAction)
        assert valid_action.tool_call_metadata.tool_call_id == 'tc-valid'

        error_action = actions[1]
        assert isinstance(error_action, FunctionCallNotExistsAction)
        assert 'not registered' in error_action.error_message
        assert error_action.tool_call_metadata.tool_call_id == 'tc-invalid'


class TestLocAgentInvalidTool:
    """B5: loc_agent now catches FunctionCallNotExistsError and returns MessageAction/FunctionCallNotExistsAction."""

    def test_returns_action_with_metadata(self):
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {}}],
        )
        actions = loc_response_to_actions(response)
        assert len(actions) == 1
        action = actions[0]
        # loc_agent may still use MessageAction or have switched to FunctionCallNotExistsAction
        assert isinstance(action, (MessageAction, FunctionCallNotExistsAction))
        assert action.tool_call_metadata is not None
        assert action.tool_call_metadata.tool_call_id == 'tc-1'


# =========================================================================
# C. End-to-end conversation structure
# =========================================================================


class TestFullEventSequence:
    """C1: Full event flow through process_events — correct message structure."""

    def test_end_to_end_structure(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()

        error_action = MessageAction(
            content='Tool bash is not registered. (arguments: {\'command\': \'ls\'}). Please check the tool name.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = _make_tool_call_metadata(
            tool_call_id='tc-1',
            function_name='bash',
        )

        events = [sys_msg, user_msg, error_action]
        messages = conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )

        # Expected: [system, user, <optional assistant thought>, user(error)]
        assert messages[0].role == 'system'
        assert messages[1].role == 'user'

        # The error must be in a user message, never assistant
        error_msg = None
        for m in messages:
            text = m.content[0].text if m.content and hasattr(m.content[0], 'text') else ''
            if 'not registered' in text:
                error_msg = m
                break
        assert error_msg is not None, 'Error message not found'
        assert error_msg.role == 'user', f'Error should be role=user, got {error_msg.role}'


class TestToolErrorIncrementsMessageCount:
    """C2: When a tool error occurs, the message count goes from N to N+1.

    Simulates two consecutive agent steps:
      Step N:  [system, user]  -> LLM called with N messages
      Step N+1: [system, user, error_action] -> LLM called with N+1 messages
    The extra message is the role='user' error feedback.
    """

    def test_message_count_increases_by_one(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()

        # Step N: baseline — only system + user
        events_step_n = [sys_msg, user_msg]
        messages_step_n = conversation_memory.process_events(
            condensed_history=events_step_n,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )
        count_n = len(messages_step_n)

        # Step N+1: tool error added to history
        error_action = MessageAction(
            content='Tool bash is not registered. (arguments: {\'command\': \'ls\'}).',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = _make_tool_call_metadata(
            tool_call_id='tc-1',
            function_name='bash',
        )

        events_step_n1 = [sys_msg, user_msg, error_action]
        messages_step_n1 = conversation_memory.process_events(
            condensed_history=events_step_n1,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )
        count_n1 = len(messages_step_n1)

        assert count_n1 == count_n + 1, (
            f'Expected message count to increase by 1 '
            f'(from {count_n} to {count_n + 1}), got {count_n1}'
        )
        # The extra message must be the user error
        extra_msg = messages_step_n1[-1]
        assert extra_msg.role == 'user'
        assert 'not registered' in extra_msg.content[0].text

    def test_message_count_with_thought_increases_by_two(self, conversation_memory):
        """When the assistant had a thought, both the thought (assistant) and the error (user)
        are added, so message count goes from N to N+2."""
        sys_msg, user_msg, initial_user = _system_and_user_events()

        # Step N: baseline
        events_step_n = [sys_msg, user_msg]
        messages_step_n = conversation_memory.process_events(
            condensed_history=events_step_n,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )
        count_n = len(messages_step_n)

        # Step N+1: tool error with assistant thought in the model_response
        response_id = 'resp-thought'
        tc_list = [
            {
                'id': 'tc-1',
                'type': 'function',
                'function': {'name': 'bash', 'arguments': '{}'},
            }
        ]
        mock_response = {
            'id': response_id,
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'Let me check the file.',
                        'tool_calls': tc_list,
                    }
                }
            ],
            'created': 0,
            'model': 'mock_model',
            'object': 'chat.completion',
            'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
        }

        error_action = MessageAction(
            content='Tool bash is not registered.',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = ToolCallMetadata(
            tool_call_id='tc-1',
            function_name='bash',
            model_response=mock_response,
            total_calls_in_response=1,
        )

        events_step_n1 = [sys_msg, user_msg, error_action]
        messages_step_n1 = conversation_memory.process_events(
            condensed_history=events_step_n1,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )
        count_n1 = len(messages_step_n1)

        # +2: assistant thought + user error
        assert count_n1 == count_n + 2, (
            f'Expected count to increase by 2 (thought + error), '
            f'from {count_n} to {count_n + 2}, got {count_n1}'
        )
        # Second-to-last is the assistant thought (tool_calls stripped)
        assert messages_step_n1[-2].role == 'assistant'
        assert 'Let me check the file' in messages_step_n1[-2].content[0].text
        # Last is the user error
        assert messages_step_n1[-1].role == 'user'
        assert 'not registered' in messages_step_n1[-1].content[0].text


class TestToolErrorInLoggedMessages:
    """C3: The messages passed to nemo_gym_client._log_completion contain the error
    as role='user', verifying it would appear correctly in logged completions.

    This simulates the flow: process_events -> messages -> model_dump() (as in logging).
    """

    def test_serialized_messages_contain_user_error(self, conversation_memory):
        sys_msg, user_msg, initial_user = _system_and_user_events()

        error_action = MessageAction(
            content='Tool bash is not registered. (arguments: {\'command\': \'ls\'}).',
            wait_for_response=False,
        )
        error_action._source = EventSource.AGENT
        error_action.tool_call_metadata = _make_tool_call_metadata(
            tool_call_id='tc-1',
            function_name='bash',
        )

        events = [sys_msg, user_msg, error_action]
        messages = conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user,
            max_message_chars=None,
            vision_is_active=False,
        )

        # Simulate what nemo_gym_client._log_completion does: serialize messages
        serialized = [m.model_dump() for m in messages]

        def _content_text(entry: dict) -> str:
            """Extract text from serialized content (may be str or list)."""
            c = entry.get('content', '')
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return ' '.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in c
                )
            return str(c)

        # Find the error in the serialized output
        error_entries = [
            d for d in serialized
            if d.get('role') == 'user' and 'not registered' in _content_text(d)
        ]
        assert len(error_entries) == 1, (
            f'Expected exactly 1 serialized user error entry, found {len(error_entries)}. '
            f'Serialized roles: {[d["role"] for d in serialized]}'
        )

        # Must NOT appear as assistant in serialized output
        assistant_error_entries = [
            d for d in serialized
            if d.get('role') == 'assistant' and 'not registered' in _content_text(d)
        ]
        assert len(assistant_error_entries) == 0, (
            'Error must not appear as assistant in serialized messages'
        )


class TestFilterUnmatchedPreservesContent:
    """C2: _filter_unmatched_tool_calls preserves assistant content when tool_calls stripped."""

    def test_assistant_content_kept_when_all_tool_calls_unmatched(self):
        from litellm import ChatCompletionMessageToolCall
        from litellm.types.utils import Function

        assistant_msg = Message(
            role='assistant',
            content=[TextContent(text='Let me check the file.')],
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id='tc-bad',
                    type='function',
                    function=Function(name='bash', arguments='{}'),
                ),
            ],
        )
        user_error = Message(
            role='user',
            content=[TextContent(text='Tool bash is not registered.')],
        )

        messages = [assistant_msg, user_error]
        filtered = list(ConversationMemory._filter_unmatched_tool_calls(messages))

        assert len(filtered) == 2
        # Assistant message kept with content, tool_calls stripped
        assert filtered[0].role == 'assistant'
        assert filtered[0].content[0].text == 'Let me check the file.'
        assert filtered[0].tool_calls is None
        # User error kept
        assert filtered[1].role == 'user'
        assert 'not registered' in filtered[1].content[0].text

    def test_assistant_dropped_when_no_content_and_all_unmatched(self):
        from litellm import ChatCompletionMessageToolCall
        from litellm.types.utils import Function

        assistant_msg = Message(
            role='assistant',
            content=[],
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id='tc-bad',
                    type='function',
                    function=Function(name='bash', arguments='{}'),
                ),
            ],
        )
        user_error = Message(
            role='user',
            content=[TextContent(text='Tool bash is not registered.')],
        )

        messages = [assistant_msg, user_error]
        filtered = list(ConversationMemory._filter_unmatched_tool_calls(messages))

        # Assistant with no content and no matched tool_calls is dropped
        assert len(filtered) == 1
        assert filtered[0].role == 'user'

    def test_partial_match_keeps_only_matched_tool_calls(self):
        from litellm import ChatCompletionMessageToolCall
        from litellm.types.utils import Function

        assistant_msg = Message(
            role='assistant',
            content=[],
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id='tc-good',
                    type='function',
                    function=Function(name='execute_bash', arguments='{}'),
                ),
                ChatCompletionMessageToolCall(
                    id='tc-bad',
                    type='function',
                    function=Function(name='bash', arguments='{}'),
                ),
            ],
        )
        tool_response = Message(
            role='tool',
            content=[TextContent(text='file1.py')],
            tool_call_id='tc-good',
            name='execute_bash',
        )
        user_error = Message(
            role='user',
            content=[TextContent(text='Tool bash is not registered.')],
        )

        messages = [assistant_msg, tool_response, user_error]
        filtered = list(ConversationMemory._filter_unmatched_tool_calls(messages))

        assert len(filtered) == 3
        # Assistant kept with only the matched tool_call
        assert filtered[0].role == 'assistant'
        assert len(filtered[0].tool_calls) == 1
        assert filtered[0].tool_calls[0].id == 'tc-good'
        # Tool response kept
        assert filtered[1].role == 'tool'
        # User error kept
        assert filtered[2].role == 'user'
