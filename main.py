from typing import List

from dotenv import load_dotenv

load_dotenv()
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, MessageGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2


def count_revisions(state: List[BaseMessage]) -> int:
    """Count completed revision rounds (the draft is not a revision)."""
    tool_turns = sum(isinstance(m, AIMessage) and bool(m.tool_calls) for m in state)
    return max(tool_turns - 1, 0)


def event_loop(state: List[BaseMessage]) -> str:
    """Stop after MAX_ITERATIONS revisions, otherwise search again."""
    if count_revisions(state) >= MAX_ITERATIONS:
        return END
    return "execute_tools"


def create_graph():
    """Create and compile the reflexion agent graph."""
    builder = MessageGraph()
    builder.add_node("draft", first_responder)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("revise", revisor)
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")
    builder.add_conditional_edges(
        "revise", event_loop, {END: END, "execute_tools": "execute_tools"}
    )
    builder.set_entry_point("draft")
    return builder.compile()


graph = create_graph()

if __name__ == "__main__":
    res = graph.invoke(
        "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
    print(res)
