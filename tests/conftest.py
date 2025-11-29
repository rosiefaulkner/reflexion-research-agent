"""Pytest configuration and shared fixtures."""

from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from schemas import AnswerQuestion, Reflection, ReviseAnswer


@pytest.fixture
def mock_tavily_tool():
    """Mock TavilySearch tool."""
    mock_tool = Mock()
    mock_tool.batch.return_value = [
        {"content": "Mock search result 1", "url": "https://example.com/1"},
        {"content": "Mock search result 2", "url": "https://example.com/2"},
    ]
    return mock_tool


@pytest.fixture
def sample_answer_question():
    """Sample AnswerQuestion instance."""
    return AnswerQuestion(
        answer="This is a test answer about AI-powered SOC.",
        reflection=Reflection(
            missing="Need more information about funding.",
            superfluous="Some unnecessary details.",
        ),
        search_queries=["AI SOC startups", "SOC funding rounds"],
    )


@pytest.fixture
def sample_revise_answer():
    """Sample ReviseAnswer instance."""
    return ReviseAnswer(
        answer="Revised answer with more details about AI-powered SOC.",
        reflection=Reflection(
            missing="Additional market data needed.",
            superfluous="Redundant information removed.",
        ),
        search_queries=["AI SOC market size", "SOC startup valuations"],
        references=["https://example.com/ref1", "https://example.com/ref2"],
    )


@pytest.fixture
def sample_messages():
    """Sample list of messages for testing."""
    return [
        HumanMessage(content="What is AI-powered SOC?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Test answer",
                        "reflection": {"missing": "", "superfluous": ""},
                        "search_queries": ["query1", "query2"],
                    },
                    "id": "test_call_id_123",
                }
            ],
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with tool calls."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "AnswerQuestion",
                "args": {
                    "answer": "Test answer",
                    "reflection": {"missing": "info", "superfluous": "extra"},
                    "search_queries": ["test query"],
                },
                "id": "test_id_456",
            }
        ],
    )


@pytest.fixture
def mock_parser():
    """Mock parser for tool calls."""
    parser = Mock()
    parser.invoke.return_value = [
        {"id": "test_call_id_123", "args": {"search_queries": ["query1", "query2"]}}
    ]
    return parser
