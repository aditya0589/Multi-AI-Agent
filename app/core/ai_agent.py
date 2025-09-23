from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

from app.config.settings import settings

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    if llm_id in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]:
        llm = ChatGroq(model=llm_id)
    else:
        llm = ChatGoogleGenerativeAI(model=llm_id)

    tools = [TavilySearch(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    state = {"messages": [query] if isinstance(query, str) else query}

    response = agent.invoke(state)

    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response from agent."
