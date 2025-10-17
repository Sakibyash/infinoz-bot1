from fastapi import FastAPI
from pydantic import BaseModel
from mem0 import Memory
import os

# mem0 will automatically pick up the MEM0_API_KEY from the environment
m = Memory()
app = FastAPI()

class ChatMessage(BaseModel):
    user_id: str
    message: str

class AIMessage(BaseModel):
    user_id: str
    user_message: str
    ai_response: str

@app.post("/get-context")
async def get_memory_context(data: ChatMessage):
    """1. Searches mem0 for relevant context and returns an enhanced system prompt."""

    relevant_memories = m.search(
        query=data.message,
        user_id=data.user_id,
        top_k=5
    )

    memories_str = "\n".join([f"- {entry['memory']}" for entry in relevant_memories])

    system_prompt = f"""
    You are an intelligent AI Assistant that uses the provided tools when necessary.
    Answer the user's question, keeping the following user memories/context in mind.

    Retrieved User Memories:
    {memories_str}

    ---
    """

    return {"system_prompt": system_prompt}

@app.post("/add-memory")
async def add_conversation_memory(data: AIMessage):
    """2. Adds the full conversation turn (User + AI) to mem0."""

    full_conversation = f"User: {data.user_message}\nAI: {data.ai_response}"
    m.add(full_conversation, user_id=data.user_id)

    return {"status": "memory added"}
