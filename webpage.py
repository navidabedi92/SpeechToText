from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np

app = Flask(__name__)

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

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio"]
    file_path = "./uploaded_audio.mp3"
    file.save(file_path)

    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    waveform = np.array(waveform, dtype=np.float32)

    sample = {"raw": waveform, "sampling_rate": sample_rate}
    result = pipe(sample)

    return jsonify({"transcription": result["text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)