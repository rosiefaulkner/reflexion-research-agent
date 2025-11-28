from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# model = init_chat_model("google_genai:gemini-2.5-flash")

configurable_model = init_chat_model(temperature=0)

response = configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "google_genai:gemini-2.5-flash"}},
)

output_parser = StrOutputParser()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "do not include any markdown, explanation or formatting, just return the answer in a plain text."),
#     ("user", "{query}")
# ])
# chain = prompt | configurable_model | output_parser
# response = chain.invoke("What is the capital of France?")
print(response)
