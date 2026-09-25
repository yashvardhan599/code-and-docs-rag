from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agent_registory.llm import llm
from agent_registory.prompts import AGENT_SYSTEM_PROMPT
from app.services.rag import format_sources_for_tool, search


def search_knowledge(query: str) -> str:
    """Search uploaded documents/code in the knowledge base and return relevant chunks."""
    hits = search(query)
    return format_sources_for_tool(hits)


agent = create_agent(
    model=llm,
    tools=[search_knowledge],
    system_prompt=AGENT_SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def chat(message: str, *, thread_id: str = "default") -> dict[str, Any]:
    """
    Send a user message to the agent.

    Pass the same thread_id to continue the same conversation
    (InMemorySaver stores history per thread).
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result.get("messages") or []
    answer = ""
    if messages:
        last = messages[-1]
        answer = getattr(last, "content", None) or str(last)

    return {
        "answer": answer,
        "thread_id": thread_id,
        "message_count": len(messages),
    }
