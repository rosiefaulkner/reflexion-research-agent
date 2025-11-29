"""FastAPI application for the Reflexion Research Agent."""

from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from main import graph

load_dotenv()

app = FastAPI(
    title="Reflexion Research Agent API",
    description="An intelligent research assistant that generates high-quality, well-researched answers through self-reflection and iterative improvement.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    """Request model for agent invocation."""

    query: str = Field(
        ...,
        description="The research question or query to investigate",
        example="What are AI-powered SOC startups and their funding?",
    )


class AgentResponse(BaseModel):
    """Response model for agent invocation."""

    answer: str = Field(..., description="The final researched answer")
    references: Optional[List[str]] = Field(
        default=None, description="List of citation URLs if available"
    )
    messages: Optional[List[dict]] = Field(
        default=None, description="Full conversation history (for debugging)"
    )


def extract_answer_from_messages(
    messages: List[BaseMessage],
) -> tuple[str, Optional[List[str]]]:
    """Extract the final answer and references from the message history."""
    answer = ""
    references = None

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                args = tool_call.get("args", {})
                if "answer" in args:
                    answer = args["answer"]
                    if "references" in args and args["references"]:
                        references = args["references"]
                    break
            if answer:
                break

    if not answer and messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            answer = last_message.content or "No answer generated"

    return answer, references


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Reflexion Research Agent API",
        "version": "1.0.0",
        "endpoints": {
            "invoke": "/v1/agent/invoke",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """
    Invoke the reflexion research agent with a query.

    The agent will:
    1. Draft an initial answer
    2. Self-critique and identify knowledge gaps
    3. Generate search queries
    4. Research using Tavily Search
    5. Revise the answer with citations
    6. Iterate up to 2 times for improvement
    """
    try:
        messages = graph.invoke(request.query)

        answer, references = extract_answer_from_messages(messages)

        messages_dict = None
        if messages:
            messages_dict = []
            for msg in messages:
                msg_dict = {
                    "type": msg.__class__.__name__,
                    "content": msg.content if hasattr(msg, "content") else None,
                }
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                messages_dict.append(msg_dict)

        return AgentResponse(
            answer=answer,
            references=references,
            messages=messages_dict,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )
