import string
from jiwer import wer
from asr.whisper_asr import WhisperASR

asr = WhisperASR()

reference = "i want to cancel my order"
prediction = asr.transcribe("temp.wav")

reference = reference.lower().translate(str.maketrans('', '', string.punctuation))
prediction = prediction.lower().translate(str.maketrans('', '', string.punctuation))

print("Prediction:", prediction)

error = wer(reference, prediction)

print("Word Error Rate:", error)