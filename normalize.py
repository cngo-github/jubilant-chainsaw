import soundfile as sf

import librosa
import pyloudnorm as pln
from pathlib import Path

sample_rate=16000
loudness_level=-23.0
true_peak_loudness_level=-1.0
use_mono=True

input_path = "clip01.wav"
output_path = f"{input_path}_normalized.wav"

path = Path(input_path)
meter = pln.Meter(sample_rate, filter_class="DeMan")

# Load and resample audio
data, sr = librosa.load(path, sr=sample_rate, mono=use_mono)

#Loudness normalize audio
original_loudness = meter.integrated_loudness(data)

normalized_audio = pln.normalize.loudness(
    data,
    input_loudness=original_loudness,
    target_loudness=loudness_level
)

# Peak-normalize audio
peak_normalized_audio = pln.normalize.peak(normalized_audio, true_peak_loudness_level)

normalized_loudness = meter.integrated_loudness(peak_normalized_audio)

print(
    f"ℹ️ File: {path.name} | Original LUFS: {original_loudness:.2f}, "
    f"Normalized LUFS: {normalized_loudness:.2f}"
)

audio = peak_normalized_audio

try:
    sf.write(
        output_path,
        audio,
        sample_rate,
        format='WAV'
    )
    print(f"ℹ️ Successfully wrote normalized audio to: {output_path}")
except Exception as e:
    print(f"🛑 Error writing normalized audio to disk: {e}")