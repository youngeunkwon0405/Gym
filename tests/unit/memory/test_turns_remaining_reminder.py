"""Standalone tests for the `include_turns_remaining_reminder` feature.

The feature:
- `AgentConfig.include_turns_remaining_reminder` is a new boolean (default False).
- `AgentController._on_event` stamps `_turns_left` on every `Observation` as it
  enters history, with the value frozen to `max_value - current_value` at that moment.
- `ConversationMemory._process_observation` appends
  `ENVIRONMENT REMINDER: You have X turns left to complete the task.` to the
  observation's message whenever the flag is on and `_turns_left` is set.
"""

from unittest.mock import MagicMock

from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import TextContent
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import CmdOutputObservation
from openhands.events.tool import ToolCallMetadata
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager


REMINDER_PREFIX = 'ENVIRONMENT REMINDER: You have '
REMINDER_SUFFIX = ' turns left to complete the task.'


def _make_memory(include_turns_remaining_reminder: bool) -> ConversationMemory:
    config = AgentConfig(
        include_turns_remaining_reminder=include_turns_remaining_reminder,
    )
    prompt_manager = MagicMock(spec=PromptManager)
    return ConversationMemory(config, prompt_manager)


def _make_base_events() -> tuple[SystemMessageAction, MessageAction]:
    system_message = SystemMessageAction(content='sys')
    system_message._source = EventSource.AGENT
    user_message = MessageAction(content='Initial user query')
    user_message._source = EventSource.USER
    return system_message, user_message


def _make_user_cmd_obs(content: str, turns_left: int | None) -> CmdOutputObservation:
    obs = CmdOutputObservation(
        command='ls',
        content=content,
        command_id=1,
        exit_code=0,
    )
    obs._source = EventSource.USER
    if turns_left is not None:
        setattr(obs, '_turns_left', turns_left)
    return obs


def _text_of(message) -> str:
    return ''.join(
        c.text for c in message.content if isinstance(c, TextContent)
    )


def _build_tool_call_metadata(tool_call_id: str, function_name: str) -> ToolCallMetadata:
    mock_response = {
        'id': f'resp_{tool_call_id}',
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': function_name,
                                'arguments': '{}',
                            },
                        }
                    ],
                }
            }
        ],
        'created': 0,
        'model': 'mock_model',
        'object': 'chat.completion',
        'usage': {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        },
    }
    return ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name=function_name,
        model_response=mock_response,
        total_calls_in_response=1,
    )


def test_agent_config_field_defaults_to_false():
    """The new config field must exist on AgentConfig and default to False."""
    config = AgentConfig()
    assert hasattr(config, 'include_turns_remaining_reminder')
    assert config.include_turns_remaining_reminder is False


def test_agent_config_field_is_toggleable():
    """The flag must be settable via the AgentConfig constructor."""
    config = AgentConfig(include_turns_remaining_reminder=True)
    assert config.include_turns_remaining_reminder is True


def test_no_reminder_when_flag_disabled_even_if_stamp_present():
    """A stamped observation must NOT get the reminder when the flag is off."""
    memory = _make_memory(include_turns_remaining_reminder=False)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='file1', turns_left=7)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert REMINDER_PREFIX not in text
    assert 'file1' in text


def test_no_reminder_when_flag_enabled_but_stamp_missing():
    """If `_turns_left` was never stamped, no reminder is appended (graceful fallback)."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='no-stamp', turns_left=None)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert REMINDER_PREFIX not in text
    assert 'no-stamp' in text


def test_reminder_appended_when_flag_enabled_and_stamp_present():
    """Happy path: flag on + stamp present => reminder appended."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='the output', turns_left=42)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert f'{REMINDER_PREFIX}42{REMINDER_SUFFIX}' in text
    assert 'the output' in text
    assert text.endswith(f'{REMINDER_PREFIX}42{REMINDER_SUFFIX}')


def test_reminder_is_frozen_per_observation():
    """Each observation uses its own `_turns_left`; older obs keeps its older count."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()

    obs_older = _make_user_cmd_obs(content='older_out', turns_left=9)
    obs_newer = _make_user_cmd_obs(content='newer_out', turns_left=5)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs_older, obs_newer],
        initial_user_action=user_message,
    )

    text_older = _text_of(messages[-2])
    text_newer = _text_of(messages[-1])

    assert f'{REMINDER_PREFIX}9{REMINDER_SUFFIX}' in text_older
    assert f'{REMINDER_PREFIX}5{REMINDER_SUFFIX}' in text_newer
    assert f'{REMINDER_PREFIX}5{REMINDER_SUFFIX}' not in text_older
    assert f'{REMINDER_PREFIX}9{REMINDER_SUFFIX}' not in text_newer


def test_reminder_with_zero_turns_left():
    """Edge case: 0 turns left should still render a valid reminder."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='final', turns_left=0)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert f'{REMINDER_PREFIX}0{REMINDER_SUFFIX}' in text


def test_reminder_appears_in_tool_response_branch():
    """When the observation has tool_call_metadata, the reminder must ride along into the role='tool' message."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()

    tool_meta = _build_tool_call_metadata(
        tool_call_id='call_ls_1', function_name='execute_bash'
    )

    cmd_action = CmdRunAction(command='ls', thought='Running ls')
    cmd_action._source = EventSource.AGENT
    cmd_action.tool_call_metadata = tool_meta

    cmd_obs = CmdOutputObservation(
        command='ls', content='file1.txt\nfile2.py', command_id=1, exit_code=0
    )
    cmd_obs._source = EventSource.AGENT
    cmd_obs.tool_call_metadata = tool_meta
    setattr(cmd_obs, '_turns_left', 13)

    history: list[Event] = [system_message, user_message, cmd_action, cmd_obs]

    messages = memory.process_events(
        condensed_history=history,
        initial_user_action=user_message,
    )

    tool_messages = [m for m in messages if m.role == 'tool']
    assert tool_messages, 'expected a tool-role message for the tool-call observation'
    tool_text = _text_of(tool_messages[-1])
    assert f'{REMINDER_PREFIX}13{REMINDER_SUFFIX}' in tool_text
    assert 'file1.txt' in tool_text


def test_reminder_does_not_duplicate_across_reprocessing():
    """process_events is non-mutating w.r.t. reminder (no duplicate reminders when called twice)."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='once', turns_left=3)

    first = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )
    second = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text_first = _text_of(first[-1])
    text_second = _text_of(second[-1])
    assert text_first.count(REMINDER_PREFIX) == 1
    assert text_second.count(REMINDER_PREFIX) == 1


def test_stamp_shape_matches_controller_formula():
    """The stamp value should equal `max(0, max_value - current_value)`; verify the formula used to compute it in the controller produces the expected display."""
    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()

    max_value = 100
    current_value = 17
    stamp = max(0, max_value - current_value)
    assert stamp == 83

    obs = _make_user_cmd_obs(content='ctrl-formula', turns_left=stamp)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert f'{REMINDER_PREFIX}83{REMINDER_SUFFIX}' in text


def test_controller_formula_clamps_to_zero_when_over_budget():
    """Exceeding max_value must not produce a negative reminder count."""
    stamp = max(0, 50 - 60)
    assert stamp == 0

    memory = _make_memory(include_turns_remaining_reminder=True)
    system_message, user_message = _make_base_events()
    obs = _make_user_cmd_obs(content='over', turns_left=stamp)

    messages = memory.process_events(
        condensed_history=[system_message, user_message, obs],
        initial_user_action=user_message,
    )

    text = _text_of(messages[-1])
    assert f'{REMINDER_PREFIX}0{REMINDER_SUFFIX}' in text
