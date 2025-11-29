"""End-to-end integration tests."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from schemas import AnswerQuestion, Reflection, ReviseAnswer


class TestEndToEnd:
    """End-to-end tests simulating real workflow."""

    @pytest.mark.integration
    @patch("tool_executor.tavily_tool")
    @patch("chains.llm")
    def test_complete_reflexion_cycle(self, mock_llm, mock_tavily_tool):
        """Test a complete reflexion cycle from question to final answer."""
        # Setup mock LLM responses
        first_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "AI-powered SOC is a security operations center enhanced with AI.",
                        "reflection": Reflection(
                            missing="Need information about specific startups and funding",
                            superfluous="",
                        ),
                        "search_queries": ["AI SOC startups funding 2024"],
                    },
                    "id": "call_initial",
                }
            ],
        )

        revise_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReviseAnswer",
                    "args": {
                        "answer": "AI-powered SOC is a security operations center enhanced with AI. Key startups include...",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": [],
                        "references": [
                            "https://example.com/startup1",
                            "https://example.com/startup2",
                        ],
                    },
                    "id": "call_revised",
                }
            ],
        )

        # Configure mock LLM to return different responses
        def llm_side_effect(*args, **kwargs):
            if not hasattr(mock_llm, "call_count"):
                mock_llm.call_count = 0
            mock_llm.call_count += 1

            if mock_llm.call_count == 1:
                return first_response
            else:
                return revise_response

        mock_llm.bind_tools.return_value.invoke.side_effect = llm_side_effect

        # Setup mock Tavily tool
        mock_tavily_tool.batch.return_value = [
            {
                "content": "Startup X raised $10M in Series A",
                "url": "https://example.com/startup1",
            },
            {
                "content": "Startup Y raised $25M in Series B",
                "url": "https://example.com/startup2",
            },
        ]

        # Import and run the graph
        from main import graph

        result = graph.invoke("What are AI-powered SOC startups and their funding?")

        # Verify results
        assert isinstance(result, list)
        assert len(result) > 0

        # Verify LLM was called
        assert mock_llm.bind_tools.return_value.invoke.called

        # Verify Tavily was called
        assert mock_tavily_tool.batch.called

    @pytest.mark.integration
    @patch("tool_executor.tavily_tool")
    @patch("chains.llm")
    def test_error_handling_in_tool_execution(self, mock_llm, mock_tavily_tool):
        """Test error handling when tool execution fails."""
        # Setup mock LLM
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Test answer",
                        "reflection": Reflection(missing="", superfluous=""),
                        "search_queries": ["test query"],
                    },
                    "id": "call_error",
                }
            ],
        )

        # Make Tavily raise an error
        mock_tavily_tool.batch.side_effect = Exception("Tavily API error")

        from main import graph

        # Should handle error gracefully or raise it
        with pytest.raises(Exception):
            graph.invoke("Test question")

    @pytest.mark.integration
    def test_schema_validation_in_workflow(self):
        """Test that schemas are properly validated during workflow."""
        # Test AnswerQuestion validation
        valid_answer = AnswerQuestion(
            answer="Test answer",
            reflection=Reflection(missing="", superfluous=""),
            search_queries=["query1", "query2"],
        )

        assert valid_answer.answer == "Test answer"
        assert len(valid_answer.search_queries) == 2

        # Test ReviseAnswer validation
        valid_revise = ReviseAnswer(
            answer="Revised answer",
            reflection=Reflection(missing="", superfluous=""),
            search_queries=["query"],
            references=["https://example.com"],
        )

        assert valid_revise.answer == "Revised answer"
        assert len(valid_revise.references) == 1
