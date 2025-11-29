#!/bin/bash
# Script to start the FastAPI server with uv

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if not already installed
echo "Installing dependencies..."
uv pip install python-dotenv langchain "langchain-google-genai>=3.2.0" langgraph "langchain-core>=1.1.0" "langchain-tavily>=0.2.13" "langgraph-prebuilt>=1.0.5" "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0"

# Start the server
echo "Starting FastAPI server..."
uvicorn api:app --reload --host 0.0.0.0 --port 8000

