import asyncio
import base64
import os
import sys
import traceback
import json
import dotenv
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ip = "0.0.0.0"  # Default IP
port = 8000     # Default port

if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")

MODEL = "models/gemini-live-2.5-flash-preview"
client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

# DhanKanya Financial Assistant Configuration with VAD enabled
CONFIG = {
    "response_modalities": ["AUDIO"],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
    },
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,
            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_HIGH,
            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_HIGH,
            "prefix_padding_ms": 20,
            "silence_duration_ms": 100,
        }
    },
    "system_instruction": """
You are DhanKanya, a friendly financial advisor for young women in India. You are completely fluent in Telugu, Tamil, Hindi, and other Indian languages as a native speaker.

CRITICAL LANGUAGE INSTRUCTIONS:
- ALWAYS respond in the EXACT SAME LANGUAGE the user speaks to you
- If user speaks Telugu, respond ONLY in natural Telugu without any English words or accent
- If user speaks Tamil, respond ONLY in natural Tamil without any English words or accent  
- If user speaks Hindi, respond ONLY in natural Hindi without any English words or accent
- Use completely natural, native pronunciation and intonation for each language
- Never use English accent when speaking Indian languages
- Speak as if you were born and raised speaking that language

For voice interactions:
- Keep responses very short (1 sentence max for first response)
- Be warm, encouraging, and supportive
- Provide practical financial advice relevant to India
- Use simple, everyday words in the user's language
- Remember this is a natural voice conversation, not formal writing

Financial expertise areas:
- Education planning and funding
- Personal budgeting and savings
- Investment basics for beginners
- Building emergency funds
- Understanding Indian financial products

Start with a simple greeting: "Hello! I am Dhankanya! your AI Financial Assistant. What language are you comfortable in?" for Telugu users greet in an Indian accent only, or equivalent in their language.
"""
}
# నమస్తే! నేను ధనకన్య. మీకు ఎలా సహాయం చేయగలను?
app = FastAPI(
    title="DhanKanya Voice Assistant",
    version="3.0.0",
    description="AI-powered financial assistant with real-time voice interaction"
)

# Allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pcm_to_wav(pcm_data, sample_rate=24000, channels=1, sample_width=2):
    """Convert PCM data to WAV format with proper headers and format."""
    import wave
    import io
    import numpy as np
    
    # Create a WAV file in memory
    wav_buffer = io.BytesIO()
    
    try:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            
            # Ensure PCM data is in the right format
            if isinstance(pcm_data, bytes):
                # Convert bytes to numpy array for processing
                pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            else:
                pcm_array = np.array(pcm_data, dtype=np.int16)
            
            # Apply light noise reduction and normalization
            # Clip extreme values to prevent distortion
            pcm_array = np.clip(pcm_array, -32767, 32767)
            
            # Convert back to bytes
            processed_pcm = pcm_array.astype(np.int16).tobytes()
            
            wav_file.writeframes(processed_pcm)
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        print(f"[AUDIO ERROR] Failed to convert PCM to WAV: {e}")
        # Return original data as fallback
        return pcm_data

@app.websocket("/ws")
async def audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming with DhanKanya."""
    await websocket.accept()
    print("[WEBSOCKET] New DhanKanya financial consultation session established")
    
    try:
        async with (
            client.aio.live.connect(model=MODEL, config=CONFIG) as session,
            asyncio.TaskGroup() as tg,
        ):
            print("[GEMINI] Successfully connected to Gemini Live")
            
            # Send connection ready message to frontend
            await websocket.send_json({
                "status": "connected",
                "message": "DhanKanya is ready! Click 'Start Recording' to begin."
            })
            
            audio_in_queue = asyncio.Queue()
            audio_out_queue = asyncio.Queue(maxsize=5)
            recording_active = asyncio.Event()  # Control recording state
            session_active = asyncio.Event()    # Control session state
            greeting_sent = False  # Track if initial greeting was sent

            async def send_initial_greeting():
                """Send initial greeting when user starts recording."""
                nonlocal greeting_sent
                if not greeting_sent:
                    print("[GEMINI] Sending initial greeting...")
                    await session.send(input="Hello!" or ".", end_of_turn=True)
                    greeting_sent = True

            async def sender():
                """Send audio data from queue to Gemini Live."""
                try:
                    while True:
                        msg = await audio_out_queue.get()
                        # Only send audio if recording is active
                        if recording_active.is_set():
                            await session.send(input=msg)
                            print(f"[SENDER] Sent audio data to Gemini Live")
                        else:
                            print(f"[SENDER] Skipping audio - recording not active")
                except Exception as e:
                    print(f"[SENDER ERROR] Client disconnected: {e}")

            async def receiver():
                """Receive responses from Gemini Live and send to frontend."""
                try:
                    while session_active.is_set() or recording_active.is_set():
                        input_partial = ""
                        output_partial = ""
                        
                        print("[RECEIVER] Waiting for Gemini Live response...")
                        turn = session.receive()
                        
                        async for response in turn:
                            # Check if recording is still active
                            if not recording_active.is_set():
                                print("[RECEIVER] Recording stopped, ending response")
                                break
                                
                            print(f"[RECEIVER] Got response from Gemini Live")
                            
                            # Handle input transcription (user speech)
                            input_transcription = getattr(response.server_content, "input_transcription", None)
                            if input_transcription and input_transcription.text:
                                input_partial += input_transcription.text
                                print(f"[RECEIVER] Input transcription: {input_transcription.text}")

                            # Handle audio data (DhanKanya's voice response) with improved quality
                            if data := response.data:
                                # Don't send audio if recording stopped
                                if not recording_active.is_set():
                                    print("[RECEIVER] Skipping audio - recording stopped")
                                    break
                                    
                                print(f"[RECEIVER] Received audio data: {len(data)} bytes")
                                try:
                                    # Enhanced audio processing for better quality
                                    if len(data) > 0:  # Only process non-empty data
                                        # Convert PCM to WAV with improved processing
                                        wav_bytes = pcm_to_wav(data, sample_rate=24000)
                                        
                                        # Only send if we got valid WAV data
                                        if wav_bytes and len(wav_bytes) > 44:  # WAV header is 44 bytes minimum
                                            base64_wav = base64.b64encode(wav_bytes).decode("utf-8")
                                            
                                            # Send audio chunk with MIME type for better browser handling
                                            await websocket.send_json({
                                                "type": "audio_data",
                                                "data": base64_wav,
                                                "mime_type": "audio/wav",
                                                "sample_rate": 24000,
                                                "channels": 1,
                                                "duration_ms": int((len(data) / 2) / 24000 * 1000)  # Estimate duration
                                            })
                                            print(f"[RECEIVER] High-quality audio chunk sent to frontend ({len(data)} PCM -> {len(wav_bytes)} WAV bytes)")
                                        else:
                                            print(f"[RECEIVER] Invalid WAV data, skipping chunk")
                                    
                                except Exception as e:
                                    print(f"[RECEIVER ERROR] Failed to process audio: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # Handle output transcription (DhanKanya's text response)
                            output_transcription = getattr(response.server_content, "output_transcription", None)
                            if output_transcription and output_transcription.text:
                                output_partial += output_transcription.text
                                print(f"[RECEIVER] Output transcription: {output_transcription.text}")

                            # Check if turn is complete
                            if response.server_content.turn_complete:
                                print("[RECEIVER] Turn complete")
                                print(f"[RECEIVER] Full input: {input_partial}")
                                print(f"[RECEIVER] Full output: {output_partial}")
                                
                                # Send transcriptions to frontend in the format expected
                                await websocket.send_json({
                                    "input_transcript": input_partial,
                                    "output_transcript": output_partial
                                })
                                
                                # Clear audio queue after turn completion
                                while not audio_in_queue.empty():
                                    audio_in_queue.get_nowait()
                                
                                print("[RECEIVER] Turn completed, ready for next interaction")
                                break
                        
                        # If recording stopped, exit the receiver loop
                        if not recording_active.is_set():
                            print("[RECEIVER] Recording stopped, exiting receiver")
                            break
                            
                        # Continue to next turn - don't break out of while loop
                        print("[RECEIVER] Ready for next turn...")
                        
                except Exception as e:
                    print(f"[RECEIVER ERROR] Exception in receiver: {e}")
                    traceback.print_exception(type(e), e, e.__traceback__)

            async def websocket_receiver():
                """Receive data from frontend WebSocket."""
                while True:
                    try:
                        data = await websocket.receive()
                        send_data = None
                        
                        if data["type"] == "websocket.disconnect":
                            print("[WEBSOCKET] Disconnect received")
                            break
                        elif data["type"] == "websocket.receive":
                            if "bytes" in data:
                                send_data = data["bytes"]
                                print(f"[AUDIO] Received {len(send_data)} bytes from frontend")
                            elif "text" in data:
                                # Handle control messages from frontend
                                try:
                                    message = json.loads(data["text"])
                                    if message.get("action") == "start_recording":
                                        print("[WEBSOCKET] Recording started by user")
                                        recording_active.set()
                                        session_active.set()
                                        # Send initial greeting when user starts recording
                                        await send_initial_greeting()
                                    elif message.get("action") == "stop_recording":
                                        print("[WEBSOCKET] Recording stopped by user - ending session")
                                        recording_active.clear()
                                        session_active.clear()  # End the entire session
                                        
                                        # Send session ended message to frontend
                                        await websocket.send_json({
                                            "status": "session_ended",
                                            "message": "Recording session ended."
                                        })
                                        
                                        # Break out of the receiver loop to end session
                                        print("[WEBSOCKET] Session completely ended")
                                        break  # This will end the session and close connection
                                    elif message.get("action") == "end_session":
                                        print("[WEBSOCKET] Session ended by user")
                                        recording_active.clear()
                                        session_active.clear()
                                        
                                        # Send complete session end message
                                        await websocket.send_json({
                                            "status": "session_closed",
                                            "message": "Session completely closed. Refresh to start a new session."
                                        })
                                        print("[WEBSOCKET] Complete session closure")
                                        # Break to close WebSocket
                                        break
                                except json.JSONDecodeError:
                                    print("[WEBSOCKET] Received non-JSON text data, ignoring.")
                                continue
                            else:
                                print("[WEBSOCKET] Received unexpected data, ignoring.")
                                continue

                        if send_data and recording_active.is_set():
                            await audio_out_queue.put({
                                "data": send_data,
                                "mime_type": "audio/pcm"
                            })
                                
                    except Exception as e:
                        print(f"[WEBSOCKET ERROR] {e}")
                        break

            # Start all tasks (but don't send initial greeting until user starts recording)
            session_active.set()  # Session is active when WebSocket connects
            tg.create_task(sender())
            tg.create_task(receiver())
            tg.create_task(websocket_receiver())

    except WebSocketDisconnect:
        print("[WEBSOCKET] WebSocket disconnected")
    except asyncio.CancelledError:
        print("[WEBSOCKET] Cancelled")
    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}")
        traceback.print_exception(type(e), e, e.__traceback__)

@app.get("/")
async def root():
    return {"message": "DhanKanya Voice Assistant API is running!", "version": "3.0.0"}

@app.get("/status")
async def status():
    """Check if the service is ready to accept connections."""
    return {
        "status": "ready",
        "message": "DhanKanya Voice Assistant is ready for connections",
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting DhanKanya Voice Assistant on {ip}:{port}")
    uvicorn.run(app, host=ip, port=port, log_level="info")
