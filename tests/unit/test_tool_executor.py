"""Unit tests for tool_executor.py."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tool_executor import execute_tools


class TestExecuteTools:
    """Tests for execute_tools function."""

    @patch("tool_executor.tavily_tool")
    @patch("tool_executor.parser")
    def test_execute_tools_single_query(self, mock_parser, mock_tavily_tool):
        """Test execute_tools with a single search query."""
        # Setup mocks
        mock_parser.invoke.return_value = [
            {"id": "call_123", "args": {"search_queries": ["test query"]}}
        ]
        mock_tavily_tool.batch.return_value = [
            {"content": "Search result 1", "url": "https://example.com"}
        ]

        # Create test messages
        messages = [
            HumanMessage(content="Test question"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["test query"]},
                        "id": "call_123",
                    }
                ],
            ),
        ]

        # Execute
        result = execute_tools(messages)

        # Assertions
        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].tool_call_id == "call_123"
        mock_parser.invoke.assert_called_once()
        mock_tavily_tool.batch.assert_called_once()

    @patch("tool_executor.tavily_tool")
    @patch("tool_executor.parser")
    def test_execute_tools_multiple_queries(self, mock_parser, mock_tavily_tool):
        """Test execute_tools with multiple search queries."""
        # Setup mocks
        mock_parser.invoke.return_value = [
            {
                "id": "call_456",
                "args": {"search_queries": ["query1", "query2", "query3"]},
            }
        ]
        mock_tavily_tool.batch.return_value = [
            {"content": "Result 1"},
            {"content": "Result 2"},
            {"content": "Result 3"},
        ]

        messages = [
            HumanMessage(content="Test"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["query1", "query2", "query3"]},
                        "id": "call_456",
                    }
                ],
            ),
        ]

        result = execute_tools(messages)

        # Should create one ToolMessage per result
        assert len(result) == 3
        assert all(isinstance(msg, ToolMessage) for msg in result)
        assert all(msg.tool_call_id == "call_456" for msg in result)
        mock_tavily_tool.batch.assert_called_once_with(
            [{"query": "query1"}, {"query": "query2"}, {"query": "query3"}]
        )

    @patch("tool_executor.tavily_tool")
    @patch("tool_executor.parser")
    def test_execute_tools_multiple_tool_calls(self, mock_parser, mock_tavily_tool):
        """Test execute_tools with multiple tool calls."""
        mock_parser.invoke.return_value = [
            {"id": "call_1", "args": {"search_queries": ["query1"]}},
            {"id": "call_2", "args": {"search_queries": ["query2"]}},
        ]

        # Each tool call gets its own batch result
        def batch_side_effect(queries):
            # Return one result per query
            return [{"content": f"Result for {q['query']}"} for q in queries]

        mock_tavily_tool.batch.side_effect = batch_side_effect

        messages = [
            HumanMessage(content="Test"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["query1"]},
                        "id": "call_1",
                    },
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["query2"]},
                        "id": "call_2",
                    },
                ],
            ),
        ]

        result = execute_tools(messages)

        # Should process all tool calls - each gets one result
        assert len(result) == 2
        assert result[0].tool_call_id == "call_1"
        assert result[1].tool_call_id == "call_2"

    @patch("tool_executor.tavily_tool")
    @patch("tool_executor.parser")
    def test_execute_tools_uses_last_message(self, mock_parser, mock_tavily_tool):
        """Test that execute_tools uses the last message in state."""
        mock_parser.invoke.return_value = [
            {"id": "call_789", "args": {"search_queries": ["test"]}}
        ]
        mock_tavily_tool.batch.return_value = [{"content": "Result"}]

        messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["test"]},
                        "id": "call_789",
                    }
                ],
            ),
        ]

        execute_tools(messages)

        # Should invoke parser with the last message (AIMessage)
        assert mock_parser.invoke.called
        called_with = mock_parser.invoke.call_args[0][0]
        assert isinstance(called_with, AIMessage)

    @patch("tool_executor.tavily_tool")
    @patch("tool_executor.parser")
    def test_execute_tools_empty_results(self, mock_parser, mock_tavily_tool):
        """Test execute_tools when Tavily returns empty results."""
        mock_parser.invoke.return_value = [
            {"id": "call_empty", "args": {"search_queries": ["query"]}}
        ]
        mock_tavily_tool.batch.return_value = []

        messages = [
            HumanMessage(content="Test"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {"search_queries": ["query"]},
                        "id": "call_empty",
                    }
                ],
            ),
        ]

        result = execute_tools(messages)

        assert len(result) == 0
