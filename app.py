from fastapi import FastAPI
from pydantic import BaseModel
from mem0 import Memory
from mem0.config import MemConfig 
import os

# ----------------------------------------------------------------------
# 1. Initialization and Configuration
# ----------------------------------------------------------------------

# 1. Define the custom configuration
# Change: Renamed class to MemConfig based on documentation
config = MemConfig(
    # LLM configuration (for fact extraction/reasoning)
    llm={
        "provider": "openai",
        "config": {
            # Change this to 'gpt-4o', 'gpt-3.5-turbo', etc., if desired
            "model": "gpt-4o-mini", 
            "temperature": 0.1,
        }
    },
    # Embedder configuration (for vector generation/search)
    embedder={
        "provider": "openai",
        "config": {
            # Change this to 'text-embedding-3-large' if you need higher accuracy
            "model": "text-embedding-3-small"
        }
    }
)

try:
    # 2. Initialize Memory using the custom config
    m = Memory.from_config(config)
except Exception as e:
    # This warning helps debug issues with API keys or network connection after import is fixed
    print(f"Warning: Failed to initialize mem0. Ensure MEM0_API_KEY and OPENAI_API_KEY are set. Error: {e}")
    m = None

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
    # Also check if memory is initialized
    status = "ok" if m is not None else "warning"
    message = "Mem0 FastAPI service is running." if m is not None else "Mem0 service is running, but Memory initialization failed. Check environment variables."
    return {"status": status, "message": message}


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
        user_id=data.user_id
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
