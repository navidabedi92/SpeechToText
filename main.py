#pip install --upgrade pip
#pip install --upgrade transformers datasets[audio] accelerate

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import librosa
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    # The line below was added to fix the issue
    return_timestamps=True  # This tells the pipeline to expect timestamps in the output
)
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print('111111')
print(result["text"])


# file_path = "/content/FemaleVoice.mp3"

# # Load the audio file as a single-channel waveform
# waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)

# # Ensure the waveform is a 1D NumPy array
# waveform = np.array(waveform, dtype=np.float32)

# # Pass the properly formatted input to the pipeline
# sample = {"raw": waveform, "sampling_rate": sample_rate}

# # Run through the pipeline
# result = pipe(sample)

# print(result["text"])