from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# llm_reasoning = ChatOpenAI(model="gpt-4.1", temperature=0)