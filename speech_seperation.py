from speechbrain.inference.separation import SepformerSeparation as separator
import soundfile as sf
import numpy as np
import torch
from pathlib import Path

input = 'clip01.wav_normalized.wav'
output_dir = Path("separated-audio")
output_dir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ℹ️ PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"ℹ️ Loading SepFormer model on device: {device}")

# Use speechbrain/sepformer-whamr for running on a GPU and speechbrain/convtasnet-pretrain-wsj02mix for CPU
model = separator.from_hparams(source="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k", savedir='pretrained_models/ConvTasNet_Libri2Mix_sepnoisy_16k')
model_sample_rate = model.hparams.sample_rate
print(f"✅ Model loaded successfully. Expected Model Rate: {model_sample_rate} Hz")

est_sources = model.separate_file(path=input)

for i, wav in enumerate(est_sources):
    # Convert tensor back to numpy array for soundfile writing
    # Squeeze removes single-dimensional dimensions
    audio_array = wav.cpu().squeeze().numpy()

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