import soundfile as sf
import torch
from speechbrain.inference.enhancement import SpectralMaskEnhancement as enhancer
from pathlib import Path

SAMPLE_RATE = 16000

input_path = 'clip01.wav_normalized.wav'
output_dir = Path("enhanced-audio")
output_dir.mkdir(exist_ok=True)
print(f"ℹ️ Output directory created at: {output_dir.absolute()}")

device = "cpu"
print(f"ℹ️ PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"⚠️ Model running on CPU. Enhancement will be slow but functional.")

model = enhancer.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir='pretrained_models/metricgan-plus-voicebank',
    run_opts={"device": device}
)

model_sample_rate = model.hparams.sample_rate
print(f"✅ Model loaded successfully. Expected Model Rate: {model_sample_rate} Hz")


print(f"ℹ️ Enhancing file: {input_path}")
# Enhancement model outputs a single enhanced signal (a single tensor)
output_filename = Path(input_path).stem.replace('_normalized', '') + '_enhanced.wav'
enhanced_wav = model.enhance_file(input_path, output_filename)

print(f"ℹ️ Enhancement complete. Writing output file.")

# Convert tensor back to numpy array for soundfile writing
audio_array = enhanced_wav.cpu().squeeze().numpy()
output_filename = Path(input_path).stem.replace('_normalized', '') + '_enhanced.wav'
enhanced_output_path = output_dir / output_filename

try:
    sf.write(
        enhanced_output_path,
        audio_array,
        model_sample_rate,
        format='WAV'
    )
    print(f"✅ Saved enhanced audio to: {enhanced_output_path.name}")
except Exception as e:
    print(f"🛑 Error writing enhanced audio: {e}")