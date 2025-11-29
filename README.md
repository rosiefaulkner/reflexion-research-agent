# Reflexion Research Agent 🦜🕸️

An intelligent research assistant that generates high-quality, well-researched answers through self-reflection and iterative improvement. The agent critiques its own responses, identifies knowledge gaps, conducts web research, and refines its answers in an iterative loop.

## What It Does

This reflexion agent is designed to answer research questions with accuracy and depth. When given a question, it:

1. **Drafts an initial answer** using an LLM (Google Gemini)
2. **Self-critiques** the answer, identifying what's missing or superfluous
3. **Generates search queries** to fill knowledge gaps
4. **Researches** using Tavily Search API to gather relevant information
5. **Revises the answer** incorporating new findings with citations
6. **Iterates** up to 2 times to continuously improve the response

The agent is particularly effective for research tasks requiring factual accuracy, such as market research, startup information, technical explanations, and domain-specific knowledge that benefits from web search validation.

## Purpose & Goals

- **Generate accurate, well-researched answers** by combining LLM reasoning with real-time web search
- **Self-improve through reflection** by identifying and addressing knowledge gaps
- **Provide citations** for verification and further reading
- **Iterate efficiently** with a controlled loop that balances thoroughness with resource usage
- **Demonstrate advanced agent patterns** using LangGraph's graph-based control flow for complex multi-step workflows

![Logo](https://github.com/emarco177/reflexion/blob/main/graph.png)

## Features

- **Self-Reflection**: Implements sophisticated reflection mechanisms for response improvement
- **Iterative Refinement**: Uses a graph-based approach to iteratively enhance responses
- **Production-Ready**: Built with scalability and real-world applications in mind
- **Integrated Search**: Leverages Tavily search for enhanced response accuracy
- **Structured Output**: Uses Pydantic models for reliable data handling

## Architecture

The agent uses a graph-based architecture with the following components:

- **Entry Point**: `draft` node for initial response generation
- **Processing Nodes**: `execute_tools` and `revise` for refinement
- **Maximum Iterations**: 2 (configurable via `MAX_ITERATIONS`)
- **Chain Components**: First responder and revisor using Google Gemini 2.5 Flash
- **Tool Integration**: Tavily Search for web research
- **State Management**: LangGraph MessageGraph for orchestrating the workflow

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file:

```bash
GOOGLE_API_KEY=your_google_api_key_here        # Required for Gemini LLM
TAVILY_API_KEY=your_tavily_api_key_here        # Required for web search
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true                      # Optional
LANGCHAIN_PROJECT=reflexion agent               # Optional
```

> **Important Note**: If you enable tracing by setting `LANGCHAIN_TRACING_V2=true`, you must have a valid LangSmith API key set in `LANGCHAIN_API_KEY`. Without a valid API key, the application will throw an error. If you don't need tracing, simply remove or comment out these environment variables.

## Run Locally

Clone the project:

```bash
git clone <repository-url>
cd reflexion-agent
```

### Option 1: Run as API Server (Recommended)

**Using `uv` (recommended):**

```bash
# Install uv if you haven't already: https://github.com/astral-sh/uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 1: Create a virtual environment
uv venv

# Step 2: Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# On macOS/Linux, you can also use: source .venv/bin/activate

# Step 3: Install dependencies (since this is a non-package project)
uv pip install python-dotenv langchain "langchain-google-genai>=3.2.0" langgraph "langchain-core>=1.1.0" "langchain-tavily>=0.2.13" "langgraph-prebuilt>=1.0.5" "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0"

# Step 4: Start the server (run directly, not with uv run)
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Quick Start Script:**

Alternatively, use the provided script:

```bash
./start_server.sh
```

**Note:** Since this project is in non-package mode, use `uv pip install` to install dependencies directly, then run `uvicorn` normally (not with `uv run`). The `uv run` command tries to build the project, which fails in non-package mode.

**Using Poetry:**

```bash
poetry install
poetry run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Option 2: Run Directly

Run the agent directly (without API):

```bash
poetry run python main.py
```

## Development Setup

1. Get your API keys:
   - [Google AI Studio](https://makersuite.google.com/app/apikey) for Gemini API access
   - [Tavily](https://tavily.com/) for search functionality
   - [LangSmith](https://smith.langchain.com/) (optional) for tracing

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your API keys

## API Usage

### Invoke the Agent

Send a POST request to `/v1/agent/invoke` with your research query:

```bash
curl -X POST "http://localhost:8000/v1/agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are AI-powered SOC startups and their funding?"
  }'
```

**Example Response:**

```json
{
  "answer": "AI-powered SOC (Security Operations Center) startups are companies that leverage artificial intelligence to enhance cybersecurity operations...",
  "references": [
    "https://example.com/startup1",
    "https://example.com/startup2"
  ],
  "messages": [...]
}
```

### Using Postman

1. Create a new POST request
2. URL: `http://localhost:8000/v1/agent/invoke`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON):
   ```json
   {
     "query": "Your research question here"
   }
   ```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation where you can test the API directly in your browser.

## Running Tests

The project includes comprehensive unit and integration tests. See [tests/README.md](tests/README.md) for detailed information.

Run all tests:
```bash
poetry run pytest
```

Run only unit tests:
```bash
poetry run pytest tests/unit/
```

Run only integration tests:
```bash
poetry run pytest tests/integration/
```

Run with verbose output:
```bash
poetry run pytest -v
```

## Acknowledgements

This project builds upon:
- [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/) for agent control flow
- [LangChain](https://github.com/langchain-ai/langchain) for LLM interactions
- [Tavily API](https://tavily.com/) for web search capabilities

