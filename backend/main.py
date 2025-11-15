import os
import uuid
from typing import Optional, List

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from gtts import gTTS

import base64
import mimetypes
import re
import struct
from google import genai
from google.genai import types

# --- 1. Load environment variables ---
load_dotenv()

# --- 2. Initialize clients ---
# Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Google AI
google_api_key = os.environ.get("GOOGLE_AI_API_KEY")
client = genai.Client(
    api_key=google_api_key,
)

# --- 3. Initialize FastAPI App ---
app = FastAPI()

origins = [
    "http://localhost:3000",  # Default Next.js dev server address
    "http://localhost:3001",  # Alternative port
    # Add your deployed web address here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Create static directory if it doesn't exist
os.makedirs("static/audio", exist_ok=True)

# Mount static directory for public file access
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 4. Define Pydantic Models (for request/response validation) ---
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    voice_name: Optional[str] = "Puck"  # Default English male voice


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
        f.close()
    print(f"File saved to: {filepath}")
    return filepath


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",  # ChunkID
        chunk_size,  # ChunkSize (total file size - 8 bytes)
        b"WAVE",  # Format
        b"fmt ",  # Subchunk1ID
        16,  # Subchunk1Size (16 for PCM)
        1,  # AudioFormat (1 for PCM)
        num_channels,  # NumChannels
        sample_rate,  # SampleRate
        byte_rate,  # ByteRate
        block_align,  # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",  # Subchunk2ID
        data_size,  # Subchunk2Size (size of audio data)
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts:  # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass  # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass  # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def generate_audio(text, voice_name="Puck", file_name="audio"):
    """Generate audio using Google Gemini TTS with English voice"""
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
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    file_index = 0
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
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
            print(f"Saved file: {file_name}{file_extension}")
            return f"{file_name}{file_extension}"
        else:
            print(chunk.text)


def generate_text(user_message, db_history):
    """Generate text response using Gemini API in English"""
    # Format history for Gemini API
    gemini_history = [
        types.Content(
            role=msg["role"],
            parts=[
                types.Part.from_text(text=msg["content"]),
            ],
        )
        for msg in db_history
    ]

    model = "gemini-2.5-flash-lite"
    tools = [
        types.Tool(url_context=types.UrlContext()),
    ]
    generate_content_config = types.GenerateContentConfig(
        top_p=0.9,
        max_output_tokens=200,
        thinking_config=types.ThinkingConfig(
            thinking_budget=8192,
        ),
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
- You're knowledgeable and helpful

Remember: Everything you say will be converted to voice, so keep your responses clear and natural for text-to-speech.

Response format: Use markdown format so the client can render it properly. Keep responses under 200 words. Avoid list formats unless specifically requested."""
            ),
        ],
    )

    # === CALL GOOGLE AI API ===
    chat_session = client.chats.create(
        model=model, config=generate_content_config, history=gemini_history
    )
    response = chat_session.send_message(user_message)
    while response.text is None:
        print("Response is None, waiting for response...")
        response = chat_session.send_message(user_message)
    return response.text


### <<< ENDPOINT: GET /conversations >>> ###
@app.get("/conversations", response_model=List[Conversation])
async def get_conversations(
    user_id: Optional[str] = None,
    # Use Query for additional validation and metadata for params
    limit: int = Query(default=20, ge=1, le=100),  # Limit from 1 to 100
    offset: int = Query(default=0, ge=0),
):
    """
    Get conversation history with pagination.
    Sorted by newest messages first.
    """
    query = supabase.table("conversations").select("*")

    if user_id:
        query = query.eq("user_id", user_id)

    # Sort by 'created_at' descending (newest first)
    # and apply pagination
    response = (
        query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    )

    return response.data


# --- 5. Define API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Main chat endpoint - receives message and returns AI response with audio"""
    user_id = request.user_id
    user_message = request.message
    # Print message from user_id
    print(f"User ID: {user_id}")
    print(f"User Message: {user_message}")

    # === STEP A: READ CONVERSATION HISTORY FROM SUPABASE ===
    query = supabase.table("conversations").select("role, content").order("created_at")
    if user_id:
        query = query.eq("user_id", user_id)
    else:
        # If no user_id, we can handle guest users
        # Example: query = query.is_("user_id", None)
        # For simplicity, we'll get all conversations (not optimal)
        # In production, you'll need session id for guests.
        pass

    db_history = query.execute().data
    ai_reply_text = generate_text(user_message, db_history)
    print(f"AI Reply Text: {ai_reply_text}")

    if ai_reply_text is None:
        return {"reply_text": "AI could not respond due to system error", "audio_url": ""}

    # === STEPS C & D & E: CREATE AUDIO FILE AND SAVE ===
    try:
        # Configure voice from request
        # Save audio file
        filename = generate_audio(
            text=ai_reply_text,
            voice_name=request.voice_name,
            file_name=f"{uuid.uuid4()}",
        )

    except Exception as e:
        print(f"Error creating audio file with Google AI TTS: {e}")
        return {"reply_text": "Error creating audio", "audio_url": ""}

    # === STEP C (continued): SAVE NEW MESSAGES TO SUPABASE ===
    # Save user message
    supabase.table("conversations").insert(
        {"user_id": user_id, "role": "user", "content": user_message}
    ).execute()

    # Save AI message
    supabase.table("conversations").insert(
        {"user_id": user_id, "role": "model", "content": ai_reply_text}
    ).execute()

    # === STEP F: RETURN TO CLIENT ===
    # Get base URL from request to create complete public URL
    base_url = str(http_request.base_url)
    audio_public_url = f"{base_url}static/audio/{filename}"

    return ChatResponse(reply_text=ai_reply_text, audio_url=audio_public_url)


# --- Root endpoint to check if server is running ---
@app.get("/")
def read_root():
    return {"message": "VRoid AI Companion Backend is running! 🚀"}
