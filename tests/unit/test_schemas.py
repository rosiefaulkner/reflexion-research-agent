"""Unit tests for schemas.py."""

import pytest
from pydantic import ValidationError

from schemas import AnswerQuestion, Reflection, ReviseAnswer


class TestReflection:
    """Tests for Reflection model."""

    def test_reflection_creation(self):
        """Test creating a valid Reflection instance."""
        reflection = Reflection(
            missing="Missing information about funding",
            superfluous="Too much detail about history",
        )
        assert reflection.missing == "Missing information about funding"
        assert reflection.superfluous == "Too much detail about history"

    def test_reflection_empty_strings(self):
        """Test Reflection with empty strings."""
        reflection = Reflection(missing="", superfluous="")
        assert reflection.missing == ""
        assert reflection.superfluous == ""

    def test_reflection_required_fields(self):
        """Test that Reflection requires both fields."""
        with pytest.raises(ValidationError):
            Reflection(missing="test")

        with pytest.raises(ValidationError):
            Reflection(superfluous="test")


class TestAnswerQuestion:
    """Tests for AnswerQuestion model."""

    def test_answer_question_creation(self, sample_answer_question):
        """Test creating a valid AnswerQuestion instance."""
        assert isinstance(sample_answer_question.answer, str)
        assert isinstance(sample_answer_question.reflection, Reflection)
        assert isinstance(sample_answer_question.search_queries, list)
        assert len(sample_answer_question.search_queries) > 0

    def test_answer_question_required_fields(self):
        """Test that AnswerQuestion requires all fields."""
        with pytest.raises(ValidationError):
            AnswerQuestion(
                answer="test", reflection=Reflection(missing="", superfluous="")
            )

    def test_answer_question_search_queries_list(self):
        """Test that search_queries must be a list."""
        answer = AnswerQuestion(
            answer="Test answer",
            reflection=Reflection(missing="", superfluous=""),
            search_queries=["query1", "query2", "query3"],
        )
        assert len(answer.search_queries) == 3
        assert all(isinstance(q, str) for q in answer.search_queries)

    def test_answer_question_model_dump(self, sample_answer_question):
        """Test that model_dump() works correctly."""
        dumped = sample_answer_question.model_dump()
        assert "answer" in dumped
        assert "reflection" in dumped
        assert "search_queries" in dumped
        assert isinstance(dumped["reflection"], dict)


class TestReviseAnswer:
    """Tests for ReviseAnswer model."""

    def test_revise_answer_creation(self, sample_revise_answer):
        """Test creating a valid ReviseAnswer instance."""
        assert isinstance(sample_revise_answer.answer, str)
        assert isinstance(sample_revise_answer.reflection, Reflection)
        assert isinstance(sample_revise_answer.search_queries, list)
        assert isinstance(sample_revise_answer.references, list)
        assert len(sample_revise_answer.references) > 0

    def test_revise_answer_inherits_from_answer_question(self, sample_revise_answer):
        """Test that ReviseAnswer inherits from AnswerQuestion."""
        assert isinstance(sample_revise_answer, AnswerQuestion)
        assert hasattr(sample_revise_answer, "references")

    def test_revise_answer_references_list(self):
        """Test that references must be a list."""
        revise = ReviseAnswer(
            answer="Test answer",
            reflection=Reflection(missing="", superfluous=""),
            search_queries=["query1"],
            references=["https://example.com/1", "https://example.com/2"],
        )
        assert len(revise.references) == 2
        assert all(isinstance(r, str) for r in revise.references)

    def test_revise_answer_required_references(self):
        """Test that ReviseAnswer requires references field."""
        with pytest.raises(ValidationError):
            ReviseAnswer(
                answer="Test answer",
                reflection=Reflection(missing="", superfluous=""),
                search_queries=["query1"],
            )
