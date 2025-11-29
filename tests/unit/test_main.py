"""Unit tests for main.py."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from main import MAX_ITERATIONS, event_loop


class TestEventLoop:
    """Tests for event_loop conditional edge function."""

    def test_event_loop_returns_execute_tools_when_below_limit(self):
        """Test that event_loop returns 'execute_tools' when below MAX_ITERATIONS."""
        messages = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
            ToolMessage(content="Tool result 1", tool_call_id="call_1"),
        ]

        result = event_loop(messages)
        assert result == "execute_tools"

    def test_event_loop_returns_execute_tools_when_at_limit(self):
        """Test that event_loop returns 'execute_tools' when exactly at MAX_ITERATIONS (uses > not >=)."""
        messages = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
            ToolMessage(content="Tool result 1", tool_call_id="call_1"),
            ToolMessage(content="Tool result 2", tool_call_id="call_2"),
        ]

        result = event_loop(messages)
        # At exactly MAX_ITERATIONS (2), it still returns execute_tools (only > MAX_ITERATIONS returns END)
        assert result == "execute_tools"

    def test_event_loop_returns_end_when_above_limit(self):
        """Test that event_loop returns END when above MAX_ITERATIONS."""
        messages = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
            ToolMessage(content="Tool result 1", tool_call_id="call_1"),
            ToolMessage(content="Tool result 2", tool_call_id="call_2"),
            ToolMessage(content="Tool result 3", tool_call_id="call_3"),
        ]

        result = event_loop(messages)
        assert result == END

    def test_event_loop_counts_only_tool_messages(self):
        """Test that event_loop only counts ToolMessage instances."""
        messages = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
            AIMessage(content="Another response"),
            ToolMessage(content="Tool result", tool_call_id="call_1"),
        ]

        result = event_loop(messages)
        # Should return execute_tools since only 1 ToolMessage
        assert result == "execute_tools"

    def test_event_loop_with_no_tool_messages(self):
        """Test event_loop with no ToolMessages."""
        messages = [HumanMessage(content="Test"), AIMessage(content="Response")]

        result = event_loop(messages)
        # 0 tool messages, so should return execute_tools
        assert result == "execute_tools"

    def test_event_loop_with_empty_messages(self):
        """Test event_loop with empty message list."""
        messages = []

        result = event_loop(messages)
        assert result == "execute_tools"

    def test_max_iterations_constant(self):
        """Test that MAX_ITERATIONS is set correctly."""
        assert MAX_ITERATIONS == 2
