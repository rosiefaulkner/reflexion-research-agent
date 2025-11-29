from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from typing import List

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

tavily_tool = TavilySearch(max_results=5)

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    
    tool_messages = []
    
    for parsed_call in parsed_tool_calls:
        call_id = parsed_call["id"]
        search_queries = parsed_call["args"]["search_queries"]
        results = tavily_tool.batch([{"query": query} for query in search_queries])
        
        # Create ToolMessage for each result
        for result in results:
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id,
                )
            )
    
    return tool_messages

if __name__ == "__main__":
    print("Tool Executor Enter")
    
    human_message = HumanMessage(
        content="Write about AI-powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-pwered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_e352d840-22d2-4147-b4d2-94abb43b03f8"
    )
    
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.model_dump(),
                        "id": "call_e352d840-22d2-4147-b4d2-94abb43b03f8",
                    }
                ],
            ),
        ]
    )
    print(raw_res)
    
