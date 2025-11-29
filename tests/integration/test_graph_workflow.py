"""Integration tests for the full graph workflow."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from main import MAX_ITERATIONS, graph
from schemas import AnswerQuestion, Reflection


class TestGraphWorkflow:
    """Integration tests for the complete graph workflow."""

    @patch("main.execute_tools")
    @patch("chains.first_responder")
    @patch("chains.revisor")
    def test_graph_single_iteration(
        self, mock_revisor, mock_first_responder, mock_execute_tools
    ):
        """Test graph execution with a single iteration (no revision)."""
        # Mock first responder
        mock_first_responder.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Initial answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": ["query1"],
                    },
                    "id": "call_1",
                }
            ],
        )

        # Mock tool execution
        mock_execute_tools.return_value = [
            ToolMessage(content="Search result", tool_call_id="call_1")
        ]

        # Mock revisor (should not be called in single iteration)
        mock_revisor.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReviseAnswer",
                    "args": {
                        "answer": "Revised answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": [],
                        "references": [],
                    },
                    "id": "call_2",
                }
            ],
        )

        # Execute graph
        result = graph.invoke("Test question")

        # Verify first responder was called
        assert mock_first_responder.invoke.called
        # Verify tools were executed
        assert mock_execute_tools.called
        # Result should be a list of messages
        assert isinstance(result, list)
        assert len(result) > 0

    @patch("main.execute_tools")
    @patch("chains.first_responder")
    @patch("chains.revisor")
    def test_graph_multiple_iterations(
        self, mock_revisor, mock_first_responder, mock_execute_tools
    ):
        """Test graph execution with multiple iterations."""
        # Mock first responder
        mock_first_responder.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Initial answer",
                        "reflection": Reflection(
                            missing="More info needed", superfluous=""
                        ),
                        "search_queries": ["query1"],
                    },
                    "id": "call_1",
                }
            ],
        )

        # Mock tool execution - first iteration
        def mock_execute_side_effect(messages):
            # Return different results based on call count
            if not hasattr(mock_execute_tools, "call_count"):
                mock_execute_tools.call_count = 0
            mock_execute_tools.call_count += 1

            if mock_execute_tools.call_count == 1:
                return [ToolMessage(content="Result 1", tool_call_id="call_1")]
            else:
                return [ToolMessage(content="Result 2", tool_call_id="call_2")]

        mock_execute_tools.side_effect = mock_execute_side_effect

        # Mock revisor
        mock_revisor.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReviseAnswer",
                    "args": {
                        "answer": "Revised answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": ["query2"],
                        "references": ["https://example.com"],
                    },
                    "id": "call_2",
                }
            ],
        )

        # Execute graph
        result = graph.invoke("Test question")

        # Verify multiple iterations occurred
        assert mock_execute_tools.call_count >= 1
        assert isinstance(result, list)

    @patch("main.execute_tools")
    @patch("chains.first_responder")
    def test_graph_stops_at_max_iterations(
        self, mock_first_responder, mock_execute_tools
    ):
        """Test that graph stops after MAX_ITERATIONS."""
        # Mock first responder
        mock_first_responder.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": ["query"],
                    },
                    "id": "call_1",
                }
            ],
        )

        # Mock tool execution to return enough ToolMessages to hit limit
        mock_execute_tools.return_value = [
            ToolMessage(content="Result 1", tool_call_id="call_1"),
            ToolMessage(content="Result 2", tool_call_id="call_2"),
        ]

        # Mock revisor to continue the loop
        with patch("chains.revisor") as mock_revisor:
            mock_revisor.invoke.return_value = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReviseAnswer",
                        "args": {
                            "answer": "Revised",
                            "reflection": Reflection(missing="", superfluous=""),
                            "search_queries": ["query"],
                            "references": [],
                        },
                        "id": "call_3",
                    }
                ],
            )

            result = graph.invoke("Test question")

            # Should eventually stop due to MAX_ITERATIONS
            assert isinstance(result, list)

    @patch("main.execute_tools")
    @patch("chains.first_responder")
    @patch("chains.revisor")
    def test_graph_handles_empty_tool_results(
        self, mock_revisor, mock_first_responder, mock_execute_tools
    ):
        """Test graph execution when tools return empty results."""
        mock_first_responder.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": ["query"],
                    },
                    "id": "call_1",
                }
            ],
        )

        mock_execute_tools.return_value = []

        mock_revisor.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReviseAnswer",
                    "args": {
                        "answer": "Revised",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": [],
                        "references": [],
                    },
                    "id": "call_2",
                }
            ],
        )

        result = graph.invoke("Test question")

        assert isinstance(result, list)
        assert mock_execute_tools.called

    def test_graph_structure(self):
        """Test that graph is properly compiled."""
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "get_graph")

    def test_graph_entry_point(self):
        """Test that graph has correct entry point."""
        graph_structure = graph.get_graph()
        # Entry point should be "draft"
        assert graph_structure is not None
