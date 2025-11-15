import os
import uuid
from typing import Optional, List

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

import mimetypes
import struct
from google import genai
from google.genai import types

# --- 1. Load environment variables ---
load_dotenv()

# Google AI
google_api_key = os.environ.get("GOOGLE_AI_API_KEY")
if not google_api_key:
    print("WARNING: GOOGLE_AI_API_KEY not set!")
    
client = genai.Client(api_key=google_api_key) if google_api_key else None

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="VRoid AI Companion API",
    description="Backend for VRoid AI Companion with Gemini",
    version="1.0.0"
)

# --- CRITICAL: Configure CORS BEFORE other middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static/audio", exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- IN-MEMORY STORAGE ---
conversation_memory = {}  # {session_id: [messages]}

# --- 3. Define Models ---
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    voice_name: Optional[str] = "Puck"


class ChatResponse(BaseModel):
    reply_text: str
    audio_url: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    id: int
    created_at: str
    user_id: Optional[str]
    role: str
    content: str


def save_binary_file(file_name, data):
    filepath = os.path.join("static/audio", file_name)
    with open(filepath, "wb") as f:
        f.write(data)
    print(f"File saved to: {filepath}")
    return filepath


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def generate_audio(text, voice_name="Puck", file_name="audio"):
    """Generate audio using Google Gemini TTS"""
    if not client:
        raise Exception("Google AI client not initialized")
        
    model = "gemini-2.5-pro-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if (
            chunk.candidates[0].content.parts[0].inline_data
            and chunk.candidates[0].content.parts[0].inline_data.data
        ):
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
            return f"{file_name}{file_extension}"


def generate_text(user_message, history):
    """Generate text response using Gemini API"""
    if not client:
        raise Exception("Google AI client not initialized")
        
    gemini_history = [
        types.Content(
            role=msg["role"],
            parts=[types.Part.from_text(text=msg["content"])],
        )
        for msg in history
    ]

    model = "gemini-2.5-flash-lite"
    tools = [types.Tool(url_context=types.UrlContext())]
    generate_content_config = types.GenerateContentConfig(
        top_p=0.9,
        max_output_tokens=200,
        thinking_config=types.ThinkingConfig(thinking_budget=8192),
        tools=tools,
        system_instruction=[
            types.Part.from_text(
                text="""You are a friendly and helpful AI companion assistant. 

IMPORTANT: You must ALWAYS respond in English.

Your personality traits:
- You are kind, empathetic, and conversational
- You respond in a natural and friendly manner
- You maintain fluid and coherent conversations
- You adapt to the user's tone
- You give concise but complete responses
- You can be fun and expressive when appropriate

Remember: Everything you say will be converted to voice, so keep your responses clear and natural for text-to-speech.

Response format: Use markdown format. Keep responses under 200 words. Avoid list formats unless specifically requested."""
            ),
        ],
    )

    chat_session = client.chats.create(
        model=model, config=generate_content_config, history=gemini_history
    )
    response = chat_session.send_message(user_message)
    while response.text is None:
        print("Response is None, waiting for response...")
        response = chat_session.send_message(user_message)
    return response.text


### --- ENDPOINTS --- ###

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "VRoid AI Companion Backend is running! 🚀",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "conversations": "/conversations",
            "chat": "/chat (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check for Railway"""
    return {
        "status": "healthy",
        "google_ai_configured": client is not None
    }


@app.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    user_id: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get conversation history from memory (not persistent)
    """
    session_id = user_id or "default"
    
    if session_id not in conversation_memory:
        return []
    
    messages = conversation_memory[session_id]
    # Return in reverse order (newest first)
    reversed_messages = list(reversed(messages))
    
    # Apply pagination
    paginated = reversed_messages[offset:offset + limit]
    
    # Format as Conversation objects
    result = []
    for idx, msg in enumerate(paginated):
        result.append(
            Conversation(
                id=idx,
                created_at="2024-01-01T00:00:00Z",
                user_id=session_id,
                role=msg["role"],
                content=msg["content"]
            )
        )
    
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Main chat endpoint - receives message and returns AI response with audio"""
    user_id = request.user_id
    session_id = user_id or "default"
    user_message = request.message
    
    print(f"Session ID: {session_id}")
    print(f"User Message: {user_message}")

    # Check if Google AI is configured
    if not client:
        return ChatResponse(
            reply_text="Sorry, the AI service is not configured. Please contact the administrator.",
            audio_url=""
        )

    # Initialize session if doesn't exist
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Get history from memory
    history = conversation_memory[session_id]
    
    # Generate AI response
    try:
        ai_reply_text = generate_text(user_message, history)
        print(f"AI Reply Text: {ai_reply_text}")
    except Exception as e:
        print(f"Error generating text: {e}")
        return ChatResponse(
            reply_text="Sorry, I encountered an error generating a response.",
            audio_url=""
        )

    if ai_reply_text is None:
        return ChatResponse(
            reply_text="AI could not respond due to system error",
            audio_url=""
        )

    # Generate audio
    try:
        filename = generate_audio(
            text=ai_reply_text,
            voice_name=request.voice_name,
            file_name=f"{uuid.uuid4()}",
        )
    except Exception as e:
        print(f"Error creating audio file: {e}")
        # Still return text even if audio fails
        filename = None

    # Save to memory
    conversation_memory[session_id].append(
        {"role": "user", "content": user_message}
    )
    conversation_memory[session_id].append(
        {"role": "model", "content": ai_reply_text}
    )

    # Return response
    base_url = str(http_request.base_url).rstrip('/')
    audio_public_url = f"{base_url}/static/audio/{filename}" if filename else ""

    return ChatResponse(reply_text=ai_reply_text, audio_url=audio_public_url)


# Add OPTIONS handler for CORS preflight
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"message": "OK"}
