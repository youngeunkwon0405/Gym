"""Comprehensive tests for FunctionCallNotExistsAction and FunctionCallNotExistsObservation.

Covers:
  A. Action & Observation unit tests (construction, defaults, properties, types)
  B. Schema enum tests
  C. Serialization round-trip tests (event_to_dict / action_from_dict / observation_from_dict)
  D. Runtime dispatch tests (action -> observation conversion)
  E. Function calling integration tests (per-agent: codeact, codex, opencode)
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from litellm import ModelResponse

from openhands.core.schema import ActionType, ObservationType
from openhands.events.action import FunctionCallNotExistsAction, ValidationFailureAction
from openhands.events.observation import FunctionCallNotExistsObservation, ValidationFailureObservation
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS, action_from_dict
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.events.serialization.observation import (
    OBSERVATION_TYPE_TO_CLASS,
    observation_from_dict,
)


# =========================================================================
# A. Action & Observation Unit Tests
# =========================================================================


class TestFunctionCallNotExistsActionConstruction:
    """A1: Basic construction and field access."""

    def test_with_all_fields(self):
        action = FunctionCallNotExistsAction(
            function_name='bash',
            error_message='Tool bash is not registered.',
            thought='I should try another tool.',
        )
        assert action.function_name == 'bash'
        assert action.error_message == 'Tool bash is not registered.'
        assert action.thought == 'I should try another tool.'

    def test_defaults(self):
        action = FunctionCallNotExistsAction()
        assert action.function_name == ''
        assert action.error_message == ''
        assert action.thought == ''

    def test_partial_fields(self):
        action = FunctionCallNotExistsAction(function_name='nonexistent')
        assert action.function_name == 'nonexistent'
        assert action.error_message == ''
        assert action.thought == ''


class TestFunctionCallNotExistsActionType:
    """A2: The action type must be FUNCTION_CALL_NOT_EXISTS, not VALIDATION_FAILURE."""

    def test_action_type_is_function_call_not_exists(self):
        action = FunctionCallNotExistsAction()
        assert action.action == ActionType.FUNCTION_CALL_NOT_EXISTS
        assert action.action == 'function_call_not_exists'

    def test_action_type_differs_from_validation_failure(self):
        fcne = FunctionCallNotExistsAction()
        vf = ValidationFailureAction()
        assert fcne.action != vf.action


class TestFunctionCallNotExistsActionMessage:
    """A3: The message property."""

    def test_message_with_function_name(self):
        action = FunctionCallNotExistsAction(
            function_name='bash',
            error_message='Tool bash is not registered.',
        )
        msg = action.message
        assert 'bash' in msg
        assert 'does not exist' in msg

    def test_message_without_function_name(self):
        action = FunctionCallNotExistsAction(
            error_message='Unknown tool.',
        )
        msg = action.message
        assert 'does not exist' in msg
        assert 'Unknown tool.' in msg

    def test_message_empty(self):
        action = FunctionCallNotExistsAction()
        msg = action.message
        assert 'does not exist' in msg


class TestFunctionCallNotExistsObservationConstruction:
    """A4: Observation construction and fields."""

    def test_with_all_fields(self):
        obs = FunctionCallNotExistsObservation(
            content='Error content',
            function_name='bash',
            error_message='Tool bash is not registered.',
        )
        assert obs.content == 'Error content'
        assert obs.function_name == 'bash'
        assert obs.error_message == 'Tool bash is not registered.'

    def test_defaults(self):
        obs = FunctionCallNotExistsObservation(content='')
        assert obs.function_name == ''
        assert obs.error_message == ''
        assert obs.content == ''


class TestFunctionCallNotExistsObservationType:
    """A5: Observation type."""

    def test_observation_type_is_function_call_not_exists(self):
        obs = FunctionCallNotExistsObservation(content='')
        assert obs.observation == ObservationType.FUNCTION_CALL_NOT_EXISTS
        assert obs.observation == 'function_call_not_exists'

    def test_observation_type_differs_from_validation_failure(self):
        fcne = FunctionCallNotExistsObservation(content='')
        vf = ValidationFailureObservation(content='')
        assert fcne.observation != vf.observation


class TestFunctionCallNotExistsObservationMessage:
    """A6: Observation message property."""

    def test_message_returns_error_message(self):
        obs = FunctionCallNotExistsObservation(
            content='content',
            error_message='Tool bash is not registered.',
        )
        assert obs.message == 'Tool bash is not registered.'

    def test_message_empty_when_no_error(self):
        obs = FunctionCallNotExistsObservation(content='content')
        assert obs.message == ''


# =========================================================================
# B. Schema Enum Tests
# =========================================================================


class TestSchemaEnums:
    """B: ActionType and ObservationType contain the new enum values."""

    def test_action_type_enum_exists(self):
        assert hasattr(ActionType, 'FUNCTION_CALL_NOT_EXISTS')
        assert ActionType.FUNCTION_CALL_NOT_EXISTS == 'function_call_not_exists'

    def test_observation_type_enum_exists(self):
        assert hasattr(ObservationType, 'FUNCTION_CALL_NOT_EXISTS')
        assert ObservationType.FUNCTION_CALL_NOT_EXISTS == 'function_call_not_exists'

    def test_action_type_distinct_from_validation_failure(self):
        assert ActionType.FUNCTION_CALL_NOT_EXISTS != ActionType.VALIDATION_FAILURE

    def test_observation_type_distinct_from_validation_failure(self):
        assert ObservationType.FUNCTION_CALL_NOT_EXISTS != ObservationType.VALIDATION_FAILURE


# =========================================================================
# C. Serialization Round-Trip Tests
# =========================================================================


class TestActionSerialization:
    """C1: ACTION_TYPE_TO_CLASS registration and round-trip."""

    def test_registered_in_action_type_to_class(self):
        key = ActionType.FUNCTION_CALL_NOT_EXISTS
        assert key in ACTION_TYPE_TO_CLASS, (
            f'{key} not found in ACTION_TYPE_TO_CLASS. '
            f'Available keys: {list(ACTION_TYPE_TO_CLASS.keys())}'
        )
        assert ACTION_TYPE_TO_CLASS[key] is FunctionCallNotExistsAction

    def test_no_collision_with_validation_failure(self):
        vf_key = ActionType.VALIDATION_FAILURE
        fcne_key = ActionType.FUNCTION_CALL_NOT_EXISTS
        assert vf_key != fcne_key
        assert ACTION_TYPE_TO_CLASS[vf_key] is ValidationFailureAction
        assert ACTION_TYPE_TO_CLASS[fcne_key] is FunctionCallNotExistsAction

    def test_event_to_dict_round_trip(self):
        action = FunctionCallNotExistsAction(
            function_name='bash',
            error_message='Tool bash is not registered.',
            thought='Trying another tool.',
        )
        d = event_to_dict(action)

        assert d['action'] == 'function_call_not_exists'
        assert d['args']['function_name'] == 'bash'
        assert d['args']['error_message'] == 'Tool bash is not registered.'
        assert d['args']['thought'] == 'Trying another tool.'

        restored = action_from_dict(d)
        assert isinstance(restored, FunctionCallNotExistsAction)
        assert restored.function_name == 'bash'
        assert restored.error_message == 'Tool bash is not registered.'
        assert restored.thought == 'Trying another tool.'
        assert restored.action == ActionType.FUNCTION_CALL_NOT_EXISTS

    def test_action_from_dict_minimal(self):
        d = {
            'action': 'function_call_not_exists',
            'args': {},
        }
        restored = action_from_dict(d)
        assert isinstance(restored, FunctionCallNotExistsAction)
        assert restored.function_name == ''
        assert restored.error_message == ''

    def test_event_from_dict_action(self):
        d = {
            'action': 'function_call_not_exists',
            'args': {
                'function_name': 'foo',
                'error_message': 'not found',
            },
        }
        evt = event_from_dict(d)
        assert isinstance(evt, FunctionCallNotExistsAction)
        assert evt.function_name == 'foo'


class TestObservationSerialization:
    """C2: OBSERVATION_TYPE_TO_CLASS registration and round-trip."""

    def test_registered_in_observation_type_to_class(self):
        key = ObservationType.FUNCTION_CALL_NOT_EXISTS
        assert key in OBSERVATION_TYPE_TO_CLASS, (
            f'{key} not found in OBSERVATION_TYPE_TO_CLASS. '
            f'Available keys: {list(OBSERVATION_TYPE_TO_CLASS.keys())}'
        )
        assert OBSERVATION_TYPE_TO_CLASS[key] is FunctionCallNotExistsObservation

    def test_no_collision_with_validation_failure(self):
        vf_key = ObservationType.VALIDATION_FAILURE
        fcne_key = ObservationType.FUNCTION_CALL_NOT_EXISTS
        assert vf_key != fcne_key
        assert OBSERVATION_TYPE_TO_CLASS[vf_key] is ValidationFailureObservation
        assert OBSERVATION_TYPE_TO_CLASS[fcne_key] is FunctionCallNotExistsObservation

    def test_event_to_dict_round_trip(self):
        obs = FunctionCallNotExistsObservation(
            content='Error content',
            function_name='bash',
            error_message='Tool bash is not registered.',
        )
        d = event_to_dict(obs)

        assert d['observation'] == 'function_call_not_exists'
        assert d['content'] == 'Error content'
        assert d['extras']['function_name'] == 'bash'
        assert d['extras']['error_message'] == 'Tool bash is not registered.'

        restored = observation_from_dict(d)
        assert isinstance(restored, FunctionCallNotExistsObservation)
        assert restored.content == 'Error content'
        assert restored.function_name == 'bash'
        assert restored.error_message == 'Tool bash is not registered.'

    def test_observation_from_dict_minimal(self):
        d = {
            'observation': 'function_call_not_exists',
            'content': '',
            'extras': {},
        }
        restored = observation_from_dict(d)
        assert isinstance(restored, FunctionCallNotExistsObservation)
        assert restored.function_name == ''

    def test_event_from_dict_observation(self):
        d = {
            'observation': 'function_call_not_exists',
            'content': 'err',
            'extras': {
                'function_name': 'foo',
                'error_message': 'not found',
            },
        }
        evt = event_from_dict(d)
        assert isinstance(evt, FunctionCallNotExistsObservation)
        assert evt.function_name == 'foo'


# =========================================================================
# D. Runtime Dispatch Tests (Action -> Observation)
# =========================================================================


class TestRuntimeActionToObservation:
    """D: FunctionCallNotExistsAction is dispatched to FunctionCallNotExistsObservation in runtime."""

    def _make_concrete_runtime(self):
        """Create a minimal concrete subclass of Runtime for testing run_action dispatch."""
        from openhands.runtime.base import Runtime

        class _StubRuntime(Runtime):
            def connect(self): ...
            def run(self, action): ...
            def run_ipython(self, action): ...
            def read(self, action): ...
            def write(self, action): ...
            def edit(self, action): ...
            def browse(self, action): ...
            def browse_interactive(self, action): ...
            def copy_to(self, *a, **kw): ...
            def copy_from(self, *a, **kw): ...
            def list_files(self, *a, **kw): ...
            def call_tool_mcp(self, action): ...
            def get_mcp_config(self): ...

        with patch.object(_StubRuntime, '__init__', lambda self: None):
            return _StubRuntime.__new__(_StubRuntime)

    def test_base_runtime_run_action(self):
        """Test the non-runnable action path in Runtime.run_action."""
        action = FunctionCallNotExistsAction(
            function_name='bad_tool',
            error_message='Tool bad_tool is not registered.',
        )
        assert not action.runnable

        runtime = self._make_concrete_runtime()
        obs = runtime.run_action(action)

        assert isinstance(obs, FunctionCallNotExistsObservation)
        assert obs.function_name == 'bad_tool'
        assert obs.error_message == 'Tool bad_tool is not registered.'
        assert obs.content == 'Tool bad_tool is not registered.'

    def test_observation_content_matches_error_message(self):
        action = FunctionCallNotExistsAction(
            function_name='xyz',
            error_message='Tool xyz is not registered. Please check the tool name.',
        )

        runtime = self._make_concrete_runtime()
        obs = runtime.run_action(action)

        assert obs.content == action.error_message


# =========================================================================
# E. Function Calling Integration Tests (per-agent)
# =========================================================================


def _make_model_response(
    tool_calls: list[dict],
    content: str | None = None,
    response_id: str = 'resp-1',
) -> ModelResponse:
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


class TestCodeactFunctionCallNotExists:
    """E1: codeact_agent response_to_actions with non-existent tools."""

    def _import_response_to_actions(self):
        from openhands.agenthub.codeact_agent.function_calling import (
            response_to_actions,
        )
        return response_to_actions

    def test_single_invalid_tool_returns_correct_type(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent_tool', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], FunctionCallNotExistsAction)

    def test_single_invalid_tool_has_correct_fields(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'bad_tool', 'arguments': {'x': 1}}],
        )
        actions = response_to_actions(response)
        action = actions[0]
        assert action.function_name == 'bad_tool'
        assert 'not registered' in action.error_message
        assert 'bad_tool' in action.error_message

    def test_single_invalid_tool_has_tool_call_metadata(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'missing', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        assert actions[0].tool_call_metadata is not None
        assert actions[0].tool_call_metadata.tool_call_id == 'tc-1'
        assert actions[0].tool_call_metadata.function_name == 'missing'
        assert actions[0].tool_call_metadata.total_calls_in_response == 1

    def test_thought_preserved_on_first_invalid_tool(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {}}],
            content='Let me try this tool.',
        )
        actions = response_to_actions(response)
        assert actions[0].thought == 'Let me try this tool.'

    def test_thought_empty_on_non_first_invalid_tool(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-1', 'name': 'execute_bash', 'arguments': {'command': 'ls'}},
                {'id': 'tc-2', 'name': 'nonexistent', 'arguments': {}},
            ],
            content='Let me try both.',
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        error_action = actions[1]
        assert isinstance(error_action, FunctionCallNotExistsAction)
        assert error_action.thought == ''

    def test_multi_tool_mix_valid_and_invalid(self):
        from openhands.events.action import CmdRunAction

        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-valid', 'name': 'execute_bash', 'arguments': {'command': 'ls'}},
                {'id': 'tc-invalid', 'name': 'doesnt_exist', 'arguments': {}},
            ],
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert isinstance(actions[0], CmdRunAction)
        assert isinstance(actions[1], FunctionCallNotExistsAction)
        assert actions[0].tool_call_metadata.total_calls_in_response == 2
        assert actions[1].tool_call_metadata.total_calls_in_response == 2

    def test_all_invalid_tools_multi(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-1', 'name': 'fake1', 'arguments': {}},
                {'id': 'tc-2', 'name': 'fake2', 'arguments': {}},
            ],
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert all(isinstance(a, FunctionCallNotExistsAction) for a in actions)

    def test_response_id_set_on_action(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {}}],
            response_id='resp-xyz',
        )
        actions = response_to_actions(response)
        assert actions[0].response_id == 'resp-xyz'


class TestCodexFunctionCallNotExists:
    """E2: codex_agent response_to_actions with non-existent tools."""

    def _import_response_to_actions(self):
        from openhands.agenthub.codex_agent.function_calling import (
            response_to_actions,
        )
        return response_to_actions

    def test_single_invalid_tool_returns_correct_type(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'not_a_tool', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], FunctionCallNotExistsAction)

    def test_single_invalid_tool_has_correct_fields(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'bad_tool', 'arguments': {'a': 1}}],
        )
        actions = response_to_actions(response)
        action = actions[0]
        assert action.function_name == 'bad_tool'
        assert 'not registered' in action.error_message

    def test_thought_preserved_on_first_invalid_tool(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {}}],
            content='Trying this tool.',
        )
        actions = response_to_actions(response)
        assert actions[0].thought == 'Trying this tool.'

    def test_multi_tool_mix_valid_and_invalid(self):
        from openhands.events.action import CmdRunAction
        from openhands.llm.tool_names import CODEX_SHELL_COMMAND_TOOL_NAME

        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-v', 'name': CODEX_SHELL_COMMAND_TOOL_NAME, 'arguments': {'command': 'pwd'}},
                {'id': 'tc-i', 'name': 'doesnt_exist', 'arguments': {}},
            ],
        )
        actions = response_to_actions(response)
        assert len(actions) == 2
        assert isinstance(actions[0], CmdRunAction)
        assert isinstance(actions[1], FunctionCallNotExistsAction)

    def test_tool_call_metadata_set(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-99', 'name': 'phantom', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        meta = actions[0].tool_call_metadata
        assert meta is not None
        assert meta.tool_call_id == 'tc-99'
        assert meta.function_name == 'phantom'


class TestOpencodeFunctionCallNotExists:
    """E3: opencode_agent response_to_actions with non-existent tools."""

    def _import_response_to_actions(self):
        from openhands.agenthub.opencode_agent.function_calling import (
            response_to_actions,
        )
        return response_to_actions

    def test_single_invalid_tool_returns_correct_type(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'ghost_tool', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        assert len(actions) == 1
        assert isinstance(actions[0], FunctionCallNotExistsAction)

    def test_single_invalid_tool_has_correct_fields(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'missing_tool', 'arguments': {'y': 2}}],
        )
        actions = response_to_actions(response)
        action = actions[0]
        assert action.function_name == 'missing_tool'
        assert 'not registered' in action.error_message

    def test_thought_preserved_on_first_invalid_tool(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-1', 'name': 'nonexistent', 'arguments': {}}],
            content='Reading code now.',
        )
        actions = response_to_actions(response)
        assert actions[0].thought == 'Reading code now.'

    def test_thought_empty_on_non_first_invalid_tool(self):
        from openhands.llm.tool_names import BASH_TOOL_NAME

        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[
                {'id': 'tc-1', 'name': BASH_TOOL_NAME, 'arguments': {'command': 'ls'}},
                {'id': 'tc-2', 'name': 'phantom', 'arguments': {}},
            ],
            content='Let me check.',
        )
        actions = response_to_actions(response)
        error_actions = [a for a in actions if isinstance(a, FunctionCallNotExistsAction)]
        assert len(error_actions) == 1
        assert error_actions[0].thought == ''

    def test_tool_call_metadata_set(self):
        response_to_actions = self._import_response_to_actions()
        response = _make_model_response(
            tool_calls=[{'id': 'tc-42', 'name': 'no_such_tool', 'arguments': {}}],
        )
        actions = response_to_actions(response)
        meta = actions[0].tool_call_metadata
        assert meta is not None
        assert meta.tool_call_id == 'tc-42'
        assert meta.function_name == 'no_such_tool'
        assert meta.total_calls_in_response == 1


# =========================================================================
# F. Action is not runnable (inherits default)
# =========================================================================


class TestFunctionCallNotExistsActionRunnable:
    """F: FunctionCallNotExistsAction should not be runnable."""

    def test_not_runnable(self):
        action = FunctionCallNotExistsAction(
            function_name='bash',
            error_message='error',
        )
        assert not action.runnable


# =========================================================================
# G. Interaction between Action and Observation
# =========================================================================


class TestActionObservationConsistency:
    """G: Fields are correctly propagated from action to observation."""

    def test_fields_propagated(self):
        action = FunctionCallNotExistsAction(
            function_name='my_tool',
            error_message='Tool my_tool is not registered.',
        )
        obs = FunctionCallNotExistsObservation(
            content=action.error_message,
            function_name=action.function_name,
            error_message=action.error_message,
        )
        assert obs.function_name == action.function_name
        assert obs.error_message == action.error_message
        assert obs.content == action.error_message

    def test_observation_message_matches_error(self):
        error = 'Tool xyz is not registered. Please check the tool name.'
        action = FunctionCallNotExistsAction(
            function_name='xyz',
            error_message=error,
        )
        obs = FunctionCallNotExistsObservation(
            content=action.error_message,
            function_name=action.function_name,
            error_message=action.error_message,
        )
        assert obs.message == error


# =========================================================================
# H. Edge Cases
# =========================================================================


class TestEdgeCases:
    """H: Edge cases and boundary conditions."""

    def test_long_error_message(self):
        long_msg = 'x' * 10000
        action = FunctionCallNotExistsAction(
            function_name='tool',
            error_message=long_msg,
        )
        assert action.error_message == long_msg
        d = event_to_dict(action)
        restored = action_from_dict(d)
        assert restored.error_message == long_msg

    def test_special_characters_in_function_name(self):
        action = FunctionCallNotExistsAction(
            function_name='tool-name_v2.0/beta',
            error_message='Not found.',
        )
        assert action.function_name == 'tool-name_v2.0/beta'
        assert 'tool-name_v2.0/beta' in action.message

    def test_unicode_in_fields(self):
        action = FunctionCallNotExistsAction(
            function_name='工具',
            error_message='工具不存在',
        )
        d = event_to_dict(action)
        restored = action_from_dict(d)
        assert restored.function_name == '工具'
        assert restored.error_message == '工具不存在'

    def test_observation_with_empty_content(self):
        obs = FunctionCallNotExistsObservation(
            content='',
            function_name='tool',
            error_message='error',
        )
        d = event_to_dict(obs)
        restored = observation_from_dict(d)
        assert restored.content == ''
        assert restored.error_message == 'error'
