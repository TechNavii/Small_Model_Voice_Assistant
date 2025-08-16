import os
import tempfile
import asyncio
import subprocess
import json
import base64
import io
import wave
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import whisper
import requests
from pydantic import BaseModel
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS with explicit settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load Whisper model
logger.info("Loading Whisper model...")
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

class TextInput(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "vad": True,
        "lm_studio_url": LM_STUDIO_URL
    }

@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    temp_files = []
    try:
        # Log request details
        content_type = audio.content_type or ""
        filename = audio.filename or ""
        logger.info(f"STT Request - Content-Type: {content_type}, Filename: {filename}")
        
        # Read audio content
        content = await audio.read()
        logger.info(f"Received audio data: {len(content)} bytes")
        
        if len(content) == 0:
            return JSONResponse(content={"text": "", "error": "Empty audio data"})
        
        # Save uploaded audio to temporary file with appropriate extension
        if "webm" in content_type or filename.endswith(".webm"):
            suffix = ".webm"
        elif "ogg" in content_type or filename.endswith(".ogg"):
            suffix = ".ogg"  
        elif "mp4" in content_type or filename.endswith(".mp4"):
            suffix = ".mp4"
        elif "mp3" in content_type or filename.endswith(".mp3"):
            suffix = ".mp3"
        else:
            suffix = ".wav"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            temp_files.append(tmp_file_path)
            logger.info(f"Saved audio to temp file: {tmp_file_path}")
        
        # If not wav, convert to wav using ffmpeg
        if suffix != ".wav":
            wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav_path = wav_file.name
            wav_file.close()
            temp_files.append(wav_path)
            
            # Convert to wav using ffmpeg with error handling
            cmd = [
                "ffmpeg", "-i", tmp_file_path, 
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",      # Mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                "-y", wav_path
            ]
            
            logger.info(f"Converting audio with command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {process.stderr}")
                # Try alternative conversion
                cmd_alt = ["ffmpeg", "-i", tmp_file_path, "-y", wav_path]
                process = subprocess.run(cmd_alt, capture_output=True, text=True)
                if process.returncode != 0:
                    raise Exception(f"FFmpeg conversion failed: {process.stderr}")
            
            tmp_file_path = wav_path
            logger.info(f"Converted audio to WAV: {wav_path}")
        
        # Transcribe audio using Whisper
        logger.info("Transcribing audio with Whisper...")
        result = whisper_model.transcribe(tmp_file_path)
        text = result["text"].strip()
        logger.info(f"Transcription result: '{text}'")
        
        return JSONResponse(content={"text": text})
    
    except Exception as e:
        logger.error(f"Speech-to-text error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

@app.post("/api/chat")
async def chat(input_data: TextInput):
    """Send text to LM Studio and get response"""
    try:
        logger.info(f"Chat request: '{input_data.text}'")
        
        # Send request to LM Studio with timeout
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": "lmstudio-community/gemma-3-270m-it-MLX-8bit",
                "messages": [
                    {"role": "user", "content": input_data.text}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"LM Studio returned status {response.status_code}")
            raise HTTPException(status_code=response.status_code, detail="LM Studio error")
        
        # Extract response text
        data = response.json()
        ai_response = data["choices"][0]["message"]["content"]
        logger.info(f"Chat response: '{ai_response[:100]}...'")
        
        return JSONResponse(content={"response": ai_response})
    
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to LM Studio")
        raise HTTPException(status_code=503, detail="Cannot connect to LM Studio. Make sure it's running on port 1234")
    except requests.exceptions.Timeout:
        logger.error("LM Studio request timeout")
        raise HTTPException(status_code=504, detail="LM Studio request timeout")
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/text-to-speech")
async def text_to_speech(input_data: TextInput):
    """Convert text to speech using Piper"""
    try:
        logger.info(f"TTS request: '{input_data.text[:100]}...'")
        
        # Try to use Piper Python API first
        try:
            from piper.voice import PiperVoice
            import wave
            
            # Load voice model - adjust path as needed
            model_path = Path("piper_models/en_US-amy-medium.onnx")
            config_path = Path("piper_models/en_US-amy-medium.onnx.json")
            
            if not model_path.exists() or not config_path.exists():
                logger.warning(f"Piper model not found at {model_path}")
                raise FileNotFoundError("Piper voice model not found")
            
            # Load the voice
            voice = PiperVoice.load(str(model_path), config_path=str(config_path))
            
            # Create temp file for audio output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                output_path = tmp_file.name
            
            # Synthesize speech
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)   # 16-bit
                wav_file.setframerate(voice.config.sample_rate)
                
                # Generate audio - synthesize returns a generator of AudioChunk objects
                audio_bytes = bytes()
                for audio_chunk in voice.synthesize(input_data.text):
                    # AudioChunk has 'audio_int16_bytes' attribute containing the raw bytes
                    audio_bytes += audio_chunk.audio_int16_bytes
                wav_file.writeframes(audio_bytes)
            
            logger.info(f"TTS generated using Python API: {output_path}")
            
            # Return the audio file
            from starlette.background import BackgroundTask
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename="speech.wav",
                background=BackgroundTask(cleanup_file, output_path)
            )
            
        except (ImportError, FileNotFoundError) as e:
            logger.warning(f"Piper Python API not available: {e}")
            
            # Fallback to command-line piper if available
            which_piper = subprocess.run(["which", "piper"], capture_output=True, text=True)
            if which_piper.returncode != 0:
                logger.warning("Piper not found")
                raise HTTPException(status_code=503, detail="Piper TTS not installed or model not found")
            
            # Create temp file for audio output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                output_path = tmp_file.name
            
            # Use command-line Piper
            model_path = Path("piper_models/en_US-amy-medium.onnx").absolute()
            config_path = Path("piper_models/en_US-amy-medium.onnx.json").absolute()
            
            process = subprocess.run(
                [
                    "piper",
                    "--model", str(model_path),
                    "--config", str(config_path),
                    "--output_file", output_path
                ],
                input=input_data.text,
                text=True,
                capture_output=True
            )
            
            if process.returncode != 0:
                logger.error(f"Piper command failed: {process.stderr}")
                raise HTTPException(status_code=503, detail="Piper TTS failed")
            
            logger.info(f"TTS generated using command-line: {output_path}")
            
            from starlette.background import BackgroundTask
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename="speech.wav",
                background=BackgroundTask(cleanup_file, output_path)
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_file(filepath: str):
    """Clean up temporary file after response is sent"""
    await asyncio.sleep(1)
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
    except:
        pass

@app.websocket("/ws/vad")
async def websocket_vad(websocket: WebSocket):
    """Simplified WebSocket endpoint for VAD - just accumulate audio and transcribe on demand"""
    await websocket.accept()
    logger.info("VAD WebSocket connected")
    
    if whisper_model is None:
        await websocket.send_json({"error": "Whisper model not loaded. Please restart the server."})
        await websocket.close()
        return
    
    audio_buffer = []
    sample_rate = 16000
    min_audio_length = 0.5  # Minimum 0.5 seconds of audio
    
    try:
        while True:
            # Receive message from client
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            # Check for control messages
            if "text" in message:
                data = message["text"]
                
                # Check if it's a control message
                try:
                    control_msg = json.loads(data)
                    if control_msg.get("action") == "transcribe":
                        # User wants to transcribe accumulated audio
                        if len(audio_buffer) > 0:
                            logger.info(f"Manual transcription requested with {len(audio_buffer)} chunks")
                            
                            # Combine all audio
                            full_audio = np.concatenate(audio_buffer)
                            audio_length = len(full_audio) / sample_rate
                            logger.info(f"Audio length: {audio_length:.2f} seconds")
                            
                            # Limit audio length to prevent issues
                            max_audio_length = 30.0  # Maximum 30 seconds
                            if audio_length > max_audio_length:
                                logger.warning(f"Audio too long ({audio_length:.1f}s), truncating to {max_audio_length}s")
                                max_samples = int(max_audio_length * sample_rate)
                                full_audio = full_audio[:max_samples]
                                audio_length = max_audio_length
                            
                            if audio_length >= min_audio_length:
                                # Save to WAV file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                    with wave.open(tmp_file.name, 'wb') as wav_file:
                                        wav_file.setnchannels(1)
                                        wav_file.setsampwidth(2)
                                        wav_file.setframerate(sample_rate)
                                        audio_int16 = (full_audio * 32768).astype(np.int16)
                                        wav_file.writeframes(audio_int16.tobytes())
                                    
                                    # Transcribe with better parameters
                                    logger.info("Transcribing with Whisper...")
                                    result = whisper_model.transcribe(
                                        tmp_file.name,
                                        language='en',
                                        initial_prompt="",  # Don't use initial prompt
                                        temperature=0.0,    # Use greedy decoding
                                        condition_on_previous_text=False  # Don't condition on previous
                                    )
                                    text = result["text"].strip()
                                    logger.info(f"Transcription: '{text}'")
                                    
                                    # Send response
                                    if text and len(text) > 1:  # Ignore single character responses
                                        await websocket.send_json({
                                            "transcription": text,
                                            "audio_length": audio_length
                                        })
                                    else:
                                        await websocket.send_json({
                                            "error": "No clear speech detected. Please speak clearly.",
                                            "audio_length": audio_length
                                        })
                                    
                                    # Clean up
                                    os.unlink(tmp_file.name)
                            else:
                                await websocket.send_json({
                                    "error": f"Audio too short ({audio_length:.1f}s), need at least {min_audio_length}s"
                                })
                            
                            # Clear buffer
                            audio_buffer = []
                        else:
                            await websocket.send_json({"error": "No audio to transcribe"})
                        continue
                    
                    elif control_msg.get("action") == "clear":
                        # Clear the audio buffer
                        audio_buffer = []
                        await websocket.send_json({"status": "buffer_cleared"})
                        continue
                        
                except json.JSONDecodeError:
                    # Not JSON, must be base64 audio
                    pass
                
                # Decode base64 audio
                try:
                    audio_data = base64.b64decode(data)
                except Exception as e:
                    logger.error(f"Failed to decode base64: {e}")
                    continue
            else:
                continue
            
            # Convert to numpy array
            try:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_np)
                
                # Send buffer status
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                buffer_duration = total_samples / sample_rate
                await websocket.send_json({
                    "status": "recording",
                    "buffer_duration": buffer_duration
                })
                
            except Exception as e:
                logger.error(f"Failed to process audio: {e}")
                continue
                    
    except WebSocketDisconnect:
        logger.info("VAD WebSocket disconnected")
    except Exception as e:
        logger.error(f"VAD WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass

# Serve static files (HTML frontend) - MUST BE LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Gemma Voice Assistant Server")
    print("=" * 60)
    print(f"Server URL: http://localhost:8000")
    print(f"LM Studio URL: {LM_STUDIO_URL}")
    print("")
    print("Features status:")
    print(f"  Whisper STT: {'✓' if whisper_model else '✗'}")
    print("")
    print("Make sure LM Studio is running with Gemma-3 model loaded!")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")