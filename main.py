import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import librosa
import numpy as np
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
processor = AutoProcessor.from_pretrained(model_id)
    
model.to(device)

# Create pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

# Load dataset sample
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

# Process dataset sample
# result = pipe(sample)
# print('111111')
# print(result["text"])

# Process local file
file_path = "FemaleVoice.mp3"  # Relative path to the project directory

waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)

# Convert waveform to NumPy format
waveform = np.array(waveform, dtype=np.float32)

# Pass formatted input to pipeline
sample = {"raw": waveform, "sampling_rate": sample_rate}
result = pipe(sample)

output_file = "./output.txt"

# Save the text result to a file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(result["text"])
