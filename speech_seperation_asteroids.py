import soundfile as sf
import torch
import numpy as np
from asteroid.models import BaseModel
import librosa
from pathlib import Path

SAMPLE_RATE = 16000

input_path = 'clip01.wav_normalized.wav'
output_dir = Path("separated-audio")
output_dir.mkdir(exist_ok=True)
print(f"ℹ️ Output directory created at: {output_dir.absolute()}")

device = "cpu"
print(f"ℹ️ PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"⚠️ Model running on CPU (Asteroid). Separation will be slow but functional.")

model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
model.to(device)

model_sample_rate = SAMPLE_RATE
print(f"✅ Asteroid Model loaded successfully. Output Sample Rate: {model_sample_rate} Hz")

print(f"ℹ️ Loading and preparing file: {input_path}")

audio_np, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

# Convert numpy array to PyTorch tensor
# Asteroid models expect input shape (batch_size, n_samples)
audio_tensor = torch.from_numpy(audio_np).float().to(device)
audio_tensor = audio_tensor.unsqueeze(0)

print(f"ℹ️ Separating audio (Input shape: {audio_tensor.shape})")
est_sources = model(audio_tensor)

# The output is typically (batch_size, n_sources, n_samples)
# We take the first batch item [0] which contains all sources.
sources = est_sources[0]
print(f"ℹ️ Separation complete. Writing {sources.shape[0]} source files.")

for i, wav_tensor in enumerate(sources):
    audio_array = wav_tensor.detach().numpy()

    # 1. DC Offset Removal (to center the signal)
    audio_array -= np.mean(audio_array)

    # Clipping Prevention Logic
    max_val = np.abs(audio_array).max()
    if max_val >= 1.0:
        # Rescale the entire array to prevent clipping (limit peak to 0.99)
        attenuation_factor = 0.99 / max_val
        audio_array *= attenuation_factor
        print(f"⚠️ Source {i + 1} clipped! Attenuating by {(20 * np.log10(attenuation_factor)):.2f} dB to prevent distortion.")

    speaker_output_path = output_dir / f"source_{i + 1}_clean.wav"

    try:
        sf.write(
            speaker_output_path,
            audio_array,
            model_sample_rate,
            format='WAV'
        )
        print(f"✅ Saved isolated speaker audio to: {speaker_output_path.name}")
    except Exception as e:
        print(f"🛑 Error writing separated audio for speaker {i + 1}: {e}")