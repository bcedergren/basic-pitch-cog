import tempfile
import os

from cog import BasePredictor, Input, Path
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH


class Predictor(BasePredictor):
    def setup(self):
        """Load the Basic Pitch model once at container startup."""
        self.model = Model(ICASSP_2022_MODEL_PATH)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file (MP3, WAV, OGG, FLAC, M4A)"),
        onset_threshold: float = Input(
            description="Minimum energy for an onset to be detected",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        frame_threshold: float = Input(
            description="Minimum energy for a frame to be considered present",
            default=0.3,
            ge=0.0,
            le=1.0,
        ),
        minimum_note_length: float = Input(
            description="Minimum note length in milliseconds",
            default=127.7,
            ge=0.0,
        ),
        minimum_frequency: float = Input(
            description="Minimum output frequency in Hz (0 = no limit)",
            default=0.0,
            ge=0.0,
        ),
        maximum_frequency: float = Input(
            description="Maximum output frequency in Hz (0 = no limit)",
            default=0.0,
            ge=0.0,
        ),
        multiple_pitch_bends: bool = Input(
            description="Allow overlapping notes to have pitch bends",
            default=False,
        ),
    ) -> Path:
        """Run Basic Pitch audio-to-MIDI transcription."""
        min_freq = minimum_frequency if minimum_frequency > 0 else None
        max_freq = maximum_frequency if maximum_frequency > 0 else None

        _, midi_data, _ = predict(
            audio_path=str(audio_file),
            model_or_model_path=self.model,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=True,
        )

        output_path = os.path.join(tempfile.mkdtemp(), "output.mid")
        midi_data.write(output_path)
        return Path(output_path)
