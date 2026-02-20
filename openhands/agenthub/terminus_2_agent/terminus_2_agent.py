"""Terminus-2 Agent for OpenHands.

A keystroke-based terminal agent that sends raw keystrokes to a terminal session
and receives screen capture output. Uses JSON-formatted LLM responses with
analysis, plan, and commands fields.
"""

import os
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.events.action import Action

from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    TerminusJSONPlainParser,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, AgentThinkAction, MessageAction
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.runtime.plugins import PluginRequirement
from openhands.utils.prompt import PromptManager


MAX_OUTPUT_BYTES = 10000
MAX_LLM_RETRY = 3

TIMEOUT_TEMPLATE = (
    'Previous command:\n{command}\n\n'
    'The previous command timed out after {timeout_sec} seconds\n\n'
    'It is possible that the command is not yet finished executing. '
    'If that is the case, then do nothing. It is also possible that you '
    'have entered an interactive shell and should continue sending '
    'keystrokes as normal.\n\n'
    'Here is the current state of the terminal:\n\n{terminal_state}'
)

COMPLETION_CONFIRMATION = (
    'Current terminal state:\n{terminal_state}\n\n'
    'Are you sure you want to mark the task as complete? '
    'This will trigger your solution to be graded and you won\'t be able to '
    'make any further corrections. If so, include "task_complete": true '
    'in your JSON response again.'
)


class Terminus2Agent(Agent):
    VERSION = '1.0'
    """
    The Terminus-2 Agent sends raw keystrokes to a terminal session and receives
    screen capture output, using JSON-formatted LLM responses.

    Unlike function-calling agents (CodeAct, OpenCode), Terminus-2 parses the
    LLM's raw text response as JSON with fields: analysis, plan, commands, and
    optionally task_complete.

    Key features:
    - Keystroke-based terminal interaction (tmux-style)
    - JSON response parsing with auto-correction
    - Double confirmation for task completion
    - Output truncation to 10KB
    - Duration-based command timeouts
    """

    sandbox_plugins: list[PluginRequirement] = []

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.parser = TerminusJSONPlainParser()
        self._pending_completion = False
        self._conversation_messages: list[dict[str, str]] = []
        self._needs_llm_call = True

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            prompt_dir = (
                self.config.custom_prompt_dir
                if self.config.custom_prompt_dir
                else os.path.join(os.path.dirname(__file__), 'prompts')
            )

            template_overrides = {}
            if self.config.system_prompt_path:
                template_overrides['system_prompt.j2'] = self.config.system_prompt_path
            if self.config.system_prompt_long_horizon_path:
                template_overrides['system_prompt_long_horizon.j2'] = (
                    self.config.system_prompt_long_horizon_path
                )

            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename=self.config.resolved_system_prompt_filename,
                template_overrides=template_overrides if template_overrides else None,
            )

        return self._prompt_manager

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()
        self._pending_completion = False
        self._conversation_messages = []
        self._needs_llm_call = True

    def step(self, state: State) -> 'Action':
        """Performs one step of the Terminus-2 agent.

        Returns pending actions from the queue, or calls the LLM to get
        new commands when the queue is empty.
        """
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        messages = self._build_messages(condensed_history, state)

        commands, is_task_complete = self._call_llm_and_parse(messages)

        if is_task_complete:
            if self._pending_completion:
                return AgentFinishAction(thought='Task completed (confirmed)')
            else:
                self._pending_completion = True
                # Execute any accompanying commands, then ask for confirmation next round
                if not commands:
                    return AgentFinishAction(thought='Task completed (confirmed)')
        else:
            self._pending_completion = False

        if not commands:
            return AgentThinkAction(thought='No commands to execute, waiting for next input')

        for cmd in commands:
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, 60),
            )
            self.pending_actions.append(action)

        return self.pending_actions.popleft()

    def _build_messages(
        self, condensed_history: list[Event], state: State
    ) -> list[Message]:
        """Build the conversation messages from event history.

        Converts the event stream into a user/assistant message sequence:
        - System message: JSON format instructions from the prompt template
        - First user message: the instruction content (from get_instruction /
          INSTRUCTION_TEMPLATE_PATH) used directly, with initial terminal state
          appended if available. This keeps the agent consistent with CodeAct/
          Codex/OpenCode which pass the MessageAction.content through unchanged,
          so that INSTRUCTION_TEMPLATE_PATH overrides work without double-wrapping.
        - Subsequent turns: assistant = LLM JSON response, user = terminal output
        """
        messages: list[Message] = []

        system_prompt = self.prompt_manager.get_system_message()
        messages.append(Message(role='system', content=[TextContent(text=system_prompt)]))

        initial_user_msg = self._find_initial_user_message(condensed_history)
        if initial_user_msg:
            initial_terminal = self._find_initial_terminal_state(condensed_history)
            if initial_terminal:
                first_user_text = (
                    f'{initial_user_msg}\n\n'
                    f'Current terminal state:\n{initial_terminal}'
                )
            else:
                first_user_text = initial_user_msg
            messages.append(
                Message(role='user', content=[TextContent(text=first_user_text)])
            )

        batch_observations: list[str] = []
        last_timed_out = False
        last_keystrokes = ''

        for event in condensed_history:
            if isinstance(event, MessageAction):
                if event.source == EventSource.USER:
                    continue
                elif event.source == EventSource.AGENT:
                    if batch_observations:
                        terminal_output = batch_observations[-1]
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        Message(
                            role='assistant',
                            content=[TextContent(text=event.content)],
                        )
                    )

            elif isinstance(event, Terminus2CmdRunAction):
                last_keystrokes = event.keystrokes
                if event.thought:
                    if batch_observations:
                        terminal_output = batch_observations[-1]
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        Message(
                            role='assistant',
                            content=[TextContent(text=event.thought)],
                        )
                    )

            elif isinstance(event, Terminus2CmdOutputObservation):
                batch_observations.append(event.terminal_state)
                last_timed_out = event.timed_out

            elif isinstance(event, ErrorObservation):
                batch_observations.append(f'ERROR: {event.content}')

            elif isinstance(event, AgentThinkAction):
                pass

        if batch_observations:
            terminal_output = batch_observations[-1]
            user_text = self._format_terminal_output(
                terminal_output, last_timed_out, last_keystrokes
            )
            messages.append(
                Message(role='user', content=[TextContent(text=user_text)])
            )

        if self._pending_completion:
            last_terminal = batch_observations[-1] if batch_observations else ''
            confirmation = COMPLETION_CONFIRMATION.format(terminal_state=last_terminal)
            messages.append(
                Message(role='user', content=[TextContent(text=confirmation)])
            )

        return messages

    def _find_initial_user_message(self, events: list[Event]) -> str | None:
        """Find the initial user message (task instruction) from the event history."""
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event.content
        return None

    def _find_initial_terminal_state(self, events: list[Event]) -> str:
        """Find the first terminal output from the event history.

        This is used to populate the {{ terminal_state }} variable in
        user_prompt.j2 for the initial user message. If no terminal
        observation has been recorded yet, returns an empty string.
        """
        for event in events:
            if isinstance(event, Terminus2CmdOutputObservation):
                return event.terminal_state
        return ''

    def _format_terminal_output(
        self, terminal_output: str, timed_out: bool, keystrokes: str
    ) -> str:
        """Format terminal output for the next user message."""
        if timed_out:
            return TIMEOUT_TEMPLATE.format(
                command=keystrokes,
                timeout_sec=60,
                terminal_state=self._limit_output_length(terminal_output),
            )
        return self._limit_output_length(terminal_output)

    def _call_llm_and_parse(
        self, messages: list[Message]
    ) -> tuple[list[ParsedCommand], bool]:
        """Call the LLM and parse the JSON response, with retry on parse errors."""
        for attempt in range(MAX_LLM_RETRY):
            params: dict = {
                'messages': messages,
            }
            response = self.llm.completion(**params)

            response_text = response.choices[0].message.content or ''
            logger.debug(f'Terminus-2 LLM response (attempt {attempt + 1}): {response_text[:200]}...')

            messages.append(
                Message(role='assistant', content=[TextContent(text=response_text)])
            )

            result = self.parser.parse_response(response_text)

            if result.error:
                feedback = f'Previous response had parsing errors:\nERROR: {result.error}'
                if result.warning:
                    feedback += f'\nWARNINGS: {result.warning}'
                feedback += '\n\nPlease fix these issues and provide a proper JSON response.'
                logger.warning(f'Terminus-2 parse error (attempt {attempt + 1}): {result.error}')

                messages.append(
                    Message(role='user', content=[TextContent(text=feedback)])
                )
                continue

            if result.warning:
                logger.info(f'Terminus-2 parse warnings: {result.warning}')

            commands = [
                ParsedCommand(keystrokes=cmd.keystrokes, duration=min(cmd.duration, 60))
                for cmd in result.commands
            ]
            return commands, result.is_task_complete

        logger.error('Terminus-2: exhausted LLM retries due to parse errors')
        return [], False

    @staticmethod
    def _limit_output_length(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        """Limit output to specified byte length, keeping first and last portions."""
        if len(output.encode('utf-8')) <= max_bytes:
            return output

        portion_size = max_bytes // 2
        output_bytes = output.encode('utf-8')
        first_portion = output_bytes[:portion_size].decode('utf-8', errors='ignore')
        last_portion = output_bytes[-portion_size:].decode('utf-8', errors='ignore')
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode('utf-8'))
            - len(last_portion.encode('utf-8'))
        )

        return (
            f'{first_portion}\n[... output limited to {max_bytes} bytes; '
            f'{omitted_bytes} interior bytes omitted ...]\n{last_portion}'
        )
