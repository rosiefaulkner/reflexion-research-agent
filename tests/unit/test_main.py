"""Unit tests for main.py: revision-based loop control."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from main import MAX_ITERATIONS, count_revisions, event_loop

_call_counter = 0


def _ai_turn(tool_name="ReviseAnswer"):
    """AI message with a tool call, like the draft or a revise produces."""
    global _call_counter
    _call_counter += 1
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": f"call_{_call_counter}"}],
    )


def _tool_results(n):
    """One ToolMessage per search query."""
    return [ToolMessage(content=f"result {i}", tool_call_id="call_x") for i in range(n)]


class TestCountRevisions:
    def test_empty_state_has_zero_revisions(self):
        assert count_revisions([]) == 0

    def test_draft_only_has_zero_revisions(self):
        state = [HumanMessage(content="q"), _ai_turn("AnswerQuestion")]
        assert count_revisions(state) == 0

    def test_one_revise_is_one_revision(self):
        state = [
            HumanMessage(content="q"),
            _ai_turn("AnswerQuestion"),
            *_tool_results(3),
            _ai_turn(),
        ]
        assert count_revisions(state) == 1

    def test_tool_messages_are_not_counted(self):
        state = [
            HumanMessage(content="q"),
            _ai_turn("AnswerQuestion"),
            *_tool_results(10),
        ]
        assert count_revisions(state) == 0


class TestEventLoop:
    def test_continues_after_first_revision(self):
        state = [
            HumanMessage(content="q"),
            _ai_turn("AnswerQuestion"),
            *_tool_results(3),
            _ai_turn(),
        ]
        assert event_loop(state) == "execute_tools"

    def test_ends_after_max_iterations_revisions(self):
        state = [HumanMessage(content="q"), _ai_turn("AnswerQuestion")]
        for _ in range(MAX_ITERATIONS):
            state += _tool_results(3) + [_ai_turn()]
        assert event_loop(state) == END

    def test_loop_length_does_not_depend_on_query_count(self):
        """Regression: the old code ended after one revision when each round had 3 queries."""
        state = [HumanMessage(content="q"), _ai_turn("AnswerQuestion")]
        state += _tool_results(3) + [_ai_turn()]
        assert event_loop(state) == "execute_tools"
        state += _tool_results(1) + [_ai_turn()]
        assert event_loop(state) == END

    def test_max_iterations_is_two(self):
        assert MAX_ITERATIONS == 2
