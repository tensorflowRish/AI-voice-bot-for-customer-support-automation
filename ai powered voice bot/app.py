from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from asr.whisper_asr import WhisperASR
from nlp.intent_model import IntentClassifier
from response.response_mapper import generate_response
from tts.tts_engine import synthesize

import shutil
import os

app = FastAPI()

# Enable CORS (important for Swagger UI and browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components
asr = WhisperASR()
intent_model = IntentClassifier()


@app.post("/voicebot")
async def voicebot(audio: UploadFile = File(...)):

    # Temporary audio file
    audio_path = "temp.wav"

    # Save uploaded audio
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Step 1: Speech → Text
    text = asr.transcribe(audio_path)

    # Step 2: Intent prediction
    intent, confidence = intent_model.predict(text)

    # Step 3: Confidence fallback
    if confidence < 0.60:
        intent = "speak_to_agent"

    # Step 4: Generate response text
    response = generate_response(intent)

    # Step 5: Text → Speech
    audio_output = synthesize(response)

    # Optional cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # Return audio response
    return FileResponse(
        audio_output,
        media_type="audio/mpeg",
        filename="response.mp3"
    )