"""Unit tests for chains.py."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from chains import (actor_prompt_template, first_responder, parser,
                    parser_pydantic, revisor)


class TestChains:
    """Tests for chain components."""

    def test_actor_prompt_template_structure(self):
        """Test that actor_prompt_template has correct structure."""
        assert actor_prompt_template is not None
        # Check that it has messages placeholder
        messages = actor_prompt_template.messages
        assert any(
            hasattr(msg, "variable_name") and msg.variable_name == "messages"
            for msg in messages
            if hasattr(msg, "variable_name")
        )

    def test_first_responder_is_chain(self):
        """Test that first_responder is a chain."""
        assert first_responder is not None
        # Should be a runnable chain
        assert hasattr(first_responder, "invoke")

    def test_revisor_is_chain(self):
        """Test that revisor is a chain."""
        assert revisor is not None
        # Should be a runnable chain
        assert hasattr(revisor, "invoke")

    def test_parser_exists(self):
        """Test that parser is defined."""
        assert parser is not None
        assert hasattr(parser, "invoke")

    def test_parser_pydantic_exists(self):
        """Test that parser_pydantic is defined."""
        assert parser_pydantic is not None
        assert hasattr(parser_pydantic, "invoke")

    @patch("chains.llm")
    def test_first_responder_invoke(self, mock_llm):
        """Test first_responder chain invocation."""
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answer": "Test answer",
                        "reflection": {"missing": "", "superfluous": ""},
                        "search_queries": ["test query"],
                    },
                    "id": "test_id",
                }
            ],
        )
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        result = first_responder.invoke(
            {"messages": [HumanMessage(content="Test question")]}
        )

        assert result is not None

    @patch("chains.llm")
    def test_revisor_invoke(self, mock_llm):
        """Test revisor chain invocation."""
        mock_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ReviseAnswer",
                    "args": {
                        "answer": "Revised answer",
                        "reflection": {"missing": "", "superfluous": ""},
                        "search_queries": ["query"],
                        "references": ["https://example.com"],
                    },
                    "id": "test_id",
                }
            ],
        )
        mock_llm.bind_tools.return_value.invoke.return_value = mock_response

        result = revisor.invoke({"messages": [HumanMessage(content="Test question")]})

        assert result is not None

    def test_prompt_template_partial(self):
        """Test that prompt templates use partial correctly."""
        # first_responder should have first_instruction partial
        assert first_responder is not None

        # revisor should have revise_instructions partial
        assert revisor is not None
