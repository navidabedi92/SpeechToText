from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
import os
import uuid
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit

# Load model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)
processor = AutoProcessor.from_pretrained(model_id)
model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files["audio"]

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4().hex}.mp3"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    file.save(file_path)

    # Load and process audio
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    waveform = np.array(waveform, dtype=np.float32)

    sample = {"raw": waveform, "sampling_rate": sample_rate}
    result = pipe(sample)

    # Cleanup temporary files
    os.remove(file_path)

    return jsonify({"filename": unique_filename, "transcription": result["text"]})

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File size exceeds 10MB limit"}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
