# Test Suite for Reflexion Agent

This directory contains comprehensive unit and integration tests for the reflexion-agent application.

## Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and pytest configuration
├── unit/                # Unit tests for individual components
│   ├── test_schemas.py
│   ├── test_tool_executor.py
│   ├── test_chains.py
│   └── test_main.py
└── integration/         # Integration tests for workflows
    ├── test_graph_workflow.py
    └── test_end_to_end.py
```

## Running Tests

### Run all tests
```bash
poetry run pytest
```

### Run only unit tests
```bash
poetry run pytest tests/unit/
```

### Run only integration tests
```bash
poetry run pytest tests/integration/
```

### Run tests with verbose output
```bash
poetry run pytest -v
```

### Run specific test file
```bash
poetry run pytest tests/unit/test_schemas.py
```

### Run specific test
```bash
poetry run pytest tests/unit/test_schemas.py::TestReflection::test_reflection_creation
```

### Skip integration tests
```bash
poetry run pytest -m "not integration"
```

### Run with coverage
```bash
poetry run pytest --cov=. --cov-report=html
```

## Test Categories

### Unit Tests

- **test_schemas.py**: Tests for Pydantic models (Reflection, AnswerQuestion, ReviseAnswer)
- **test_tool_executor.py**: Tests for tool execution logic with mocked Tavily API
- **test_chains.py**: Tests for LangChain chain components
- **test_main.py**: Tests for graph conditional edge logic

### Integration Tests

- **test_graph_workflow.py**: Tests for complete graph execution workflows
- **test_end_to_end.py**: End-to-end tests simulating real user interactions

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `mock_tavily_tool`: Mock TavilySearch tool
- `sample_answer_question`: Sample AnswerQuestion instance
- `sample_revise_answer`: Sample ReviseAnswer instance
- `sample_messages`: Sample message list for testing
- `mock_llm_response`: Mock LLM response with tool calls
- `mock_parser`: Mock parser for tool calls

## Mocking

Tests use `unittest.mock` to mock external dependencies:
- LLM API calls (Google Gemini)
- Tavily Search API
- LangChain components

This ensures tests run quickly and don't require API keys or network access.

## Writing New Tests

When adding new tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use descriptive test class and method names
4. Add appropriate docstrings
5. Use fixtures from `conftest.py` when possible
6. Mock external dependencies

Example:
```python
def test_new_feature(self, sample_answer_question):
    """Test description."""
    result = function_under_test(sample_answer_question)
    assert result is not None
```

