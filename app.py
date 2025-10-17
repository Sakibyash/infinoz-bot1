from fastapi import FastAPI
from pydantic import BaseModel
from mem0 import Memory
import os

# ----------------------------------------------------------------------
# 1. Initialization and Configuration
# ----------------------------------------------------------------------
# The 'Memory()' initialization will automatically look for the 
# MEM0_API_KEY and OPENAI_API_KEY environment variables set in Render.
try:
    m = Memory()
except Exception as e:
    # Handle case where MEM0_API_KEY might be missing during testing
    print(f"Warning: Failed to initialize mem0. Ensure MEM0_API_KEY is set. Error: {e}")
    m = None # Keep m as None if initialization failed

app = FastAPI(
    title="Mem0 Memory Handler", 
    description="External service for n8n to manage long-term conversation memory."
)

# Define the expected JSON body structure for receiving data from n8n
class ChatMessage(BaseModel):
    user_id: str
    message: str
    
class AIMessage(BaseModel):
    user_id: str
    user_message: str
    ai_response: str


# ----------------------------------------------------------------------
# 2. Endpoints
# ----------------------------------------------------------------------

@app.get("/", summary="Health Check")
async def root():
    """Simple health check to verify the service is running."""
    return {"status": "ok", "message": "Mem0 FastAPI service is running."}


@app.post("/get-context", summary="Retrieve Relevant User Context")
async def get_memory_context(data: ChatMessage):
    """
    Called BEFORE the AI Agent. Searches mem0 for context 
    relevant to the user's latest message.
    """
    if m is None:
        return {"system_prompt": "Error: Memory service not configured."}
    
    # Search for Relevant Memories
    relevant_memories = m.search(
        query=data.message,
        user_id=data.user_id,
        top_k=5  # Adjust as needed
    )

    # Format the memories into a concise string for the AI Agent
    if relevant_memories:
        memories_str = "\n".join([f"- {entry['memory']}" for entry in relevant_memories])
        context_block = f"""
        Retrieved User Memories (USE THESE FOR CONTEXT):
        {memories_str}
        
        ---
        """
    else:
        context_block = "No specific past memories were found for this user."

    # Create the Enhanced System Prompt
    system_prompt = f"""
    You are an intelligent AI Assistant. 
    
    {context_block}

    Respond to the user's latest message: "{data.message}"
    """
    
    return {"system_prompt": system_prompt}


@app.post("/add-memory", summary="Log Conversation for Future Context")
async def add_conversation_memory(data: AIMessage):
    """
    Called AFTER the AI Agent. Logs the conversation turn to mem0 
    to build the user's long-term memory.
    """
    if m is None:
        return {"status": "error", "message": "Memory service not configured."}

    # Combine User and AI responses into a single memory entry
    full_conversation = f"User: {data.user_message}\nAI: {data.ai_response}"
    
    m.add(full_conversation, user_id=data.user_id)
    
    return {"status": "memory added successfully"}
