import librosa
import numpy as np
import pyloudnorm as pln
from pyloudnorm.meter import Meter
from pathlib import Path
import logging
from audioread import NoBackendError
from librosa import ParameterError

# Setup logging for better production management
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AudioSample:
    data: np.ndarray
    sample_rate: int

    def __init__(self, data, sample_rate: int):
        self.data = np.asarray(data)
        self.sample_rate = sample_rate

class Normalizer:
    __sample_rate: int
    __loudness_level: float
    __use_mono: bool # Monophonic sound reduces computational load and is better for speaker diarization, Whisper, and TitaNet.
    __meter: Meter

    def __init__(self, sample_rate: int=16000, loudness_level: float=-23.0, use_mono: bool=True):
        self.__sample_rate = sample_rate
        self.__loudness_level = loudness_level
        self.__use_mono = use_mono
        self.__meter = pln.Meter(sample_rate)

    def get_loudness(self, audio: AudioSample) -> float:
        return self.__meter.integrated_loudness(audio.data)

    def normalize(self, path: str | Path) -> AudioSample | None:
        path = Path(path)
        audio = self._load_and_resample(path)

        if audio is None:
            return None

        original_loudness = self.get_loudness(audio)

        normalized_audio = pln.normalize.loudness(
            audio.data,
            input_loudness=original_loudness,
            target_loudness=self.__loudness_level
        )

        normalized_audio_sample = AudioSample(normalized_audio, audio.sample_rate)
        normalized_loudness = self.get_loudness(normalized_audio_sample)

        logger.info(
            f"ℹ️ File: {path.name} | Original LUFS: {original_loudness:.2f}, "
            f"Normalized LUFS: {normalized_loudness:.2f}"
        )

        return normalized_audio_sample

    def _load_and_resample(self, path: Path) -> AudioSample | None:
        try:
            data, sr = librosa.load(path, sr=self.__sample_rate, mono=self.__use_mono)
            return AudioSample(data=data, sample_rate=sr)
        except NoBackendError as e:
            logger.error(f"🛑 Unable to process the audio: {e}")
        except FileNotFoundError:
            logger.error(f"🛑 The file {path} could not be found.")
        except (OSError, ParameterError) as e:
            logger.error(f"🛑 Error loading or resampling audio: {e}")

        return None

if __name__ == "__main__":
    import soundfile as sf

    input_path = "clip01.wav"
    output_path = f"{input_path}_normalized.wav"

    normalizer = Normalizer()
    audio = normalizer.normalize(input_path)

    if audio is not None:
        try:
            sf.write(
                output_path,
                audio.data,
                audio.sample_rate,
                format='WAV'
            )
            print(f"ℹ️ Successfully wrote normalized audio to: {output_path}")
        except Exception as e:
            print(f"🛑 Error writing normalized audio to disk: {e}")
    else:
        print(f"ℹ️ Failed to process audio {input_path}")