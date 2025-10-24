import os
import uuid
from typing import Optional, List

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Optional: Comment out these lines if you don't want to use Supabase
from supabase import create_client, Client

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. Initialize clients ---

# OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAI(api_key=openai_api_key)

# Supabase (Optional - for conversation history)
USE_SUPABASE = os.environ.get("USE_SUPABASE", "false").lower() == "true"
supabase: Optional[Client] = None

if USE_SUPABASE:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase enabled for conversation history")
    else:
        print("⚠️ Supabase credentials missing. History disabled.")
        USE_SUPABASE = False
else:
    print("ℹ️ Supabase disabled. Running without conversation history.")

# --- 3. Initialize FastAPI App ---
app = FastAPI(title="VRoid AI Companion - OpenAI Edition")

# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",  # Allow Vercel deployments
    "*",  # In production, replace with your actual domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for audio files
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 4. Pydantic Models ---
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    voice_name: Optional[str] = "alloy"  # alloy, echo, fable, onyx, nova, shimmer


class ChatResponse(BaseModel):
    reply_text: str
    audio_url: str


class Conversation(BaseModel):
    id: int
    created_at: str
    user_id: Optional[str]
    role: str
    content: str


# --- 5. Helper Functions ---

def generate_text_response(user_message: str, conversation_history: List[dict] = None) -> str:
    """
    Generate a text response using OpenAI ChatGPT
    """
    messages = [
        {
            "role": "system",
            "content": """You are a friendly AI companion avatar. Your personality traits:
- Warm, engaging, and conversational
- Keep responses concise (2-3 sentences max, under 150 words)
- Natural and expressive - show personality!
- Helpful and supportive
- Always respond in English

Remember: Your responses will be spoken out loud by a 3D avatar, so keep them conversational and natural."""
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use "gpt-4o" for better quality (more expensive)
            messages=messages,
            temperature=0.8,
            max_tokens=150,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        return "I'm having trouble thinking right now. Could you try again?"


def generate_audio(text: str, voice_name: str = "alloy") -> Optional[str]:
    """
    Generate audio using OpenAI TTS and save to file
    Returns the filename if successful, None otherwise
    """
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join("static/audio", filename)
        
        # Generate audio with OpenAI TTS
        response = openai_client.audio.speech.create(
            model="tts-1",  # Use "tts-1-hd" for higher quality
            voice=voice_name,
            input=text,
            speed=1.0
        )
        
        # Save to file
        response.stream_to_file(filepath)
        print(f"✅ Audio saved: {filename}")
        
        return filename
    
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return None


def save_to_history(user_id: Optional[str], role: str, content: str):
    """
    Save message to Supabase (if enabled)
    """
    if not USE_SUPABASE or not supabase:
        return
    
    try:
        supabase.table("conversations").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        print(f"⚠️ Error saving to history: {e}")


def get_conversation_history(user_id: Optional[str] = None) -> List[dict]:
    """
    Retrieve conversation history from Supabase (if enabled)
    """
    if not USE_SUPABASE or not supabase:
        return []
    
    try:
        query = supabase.table("conversations").select("role, content").order("created_at")
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        result = query.execute()
        return result.data
    
    except Exception as e:
        print(f"⚠️ Error retrieving history: {e}")
        return []


# --- 6. API Endpoints ---

@app.get("/")
def read_root():
    return {
        "message": "🤖 VRoid AI Companion Backend - OpenAI Edition",
        "status": "running",
        "features": {
            "model": "OpenAI GPT-4o-mini",
            "tts": "OpenAI TTS",
            "history": "Enabled" if USE_SUPABASE else "Disabled"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint - generates text and audio responses
    """
    user_id = request.user_id
    user_message = request.message.strip()
    
    print(f"💬 User ({user_id or 'guest'}): {user_message}")
    
    if not user_message:
        return ChatResponse(
            reply_text="I didn't catch that. Could you say something?",
            audio_url=""
        )
    
    # Get conversation history
    conversation_history = get_conversation_history(user_id)
    
    # Generate AI response
    ai_reply = generate_text_response(user_message, conversation_history)
    print(f"🤖 AI: {ai_reply}")
    
    # Generate audio
    audio_filename = generate_audio(ai_reply, request.voice_name)
    
    if not audio_filename:
        print("⚠️ Audio generation failed, returning text only")
        audio_url = ""
    else:
        # Create public URL for audio file
        base_url = str(http_request.base_url).rstrip('/')
        audio_url = f"{base_url}/static/audio/{audio_filename}"
    
    # Save to history
    save_to_history(user_id, "user", user_message)
    save_to_history(user_id, "assistant", ai_reply)
    
    return ChatResponse(
        reply_text=ai_reply,
        audio_url=audio_url
    )


@app.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    user_id: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Retrieve conversation history with pagination
    """
    if not USE_SUPABASE or not supabase:
        return []
    
    try:
        query = supabase.table("conversations").select("*")
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        response = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching conversations: {e}")
        return []


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "openai": "connected" if openai_api_key else "missing",
        "supabase": "connected" if USE_SUPABASE else "disabled"
    }


# --- Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000 ---