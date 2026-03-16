import tempfile
import os

from cog import BasePredictor, Input, Path, BaseModel
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH


class Output(BaseModel):
    midi_file: Path
    musicxml_file: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the Basic Pitch model once at container startup."""
        self.model = Model(ICASSP_2022_MODEL_PATH)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file (MP3, WAV, OGG, FLAC, M4A)"),
        stem_name: str = Input(
            description="Stem instrument name for notation mode (e.g. 'guitar', 'bass', 'drums', 'vocals', 'piano'). Determines clef, staff type, and notation style.",
            default="",
        ),
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
    ) -> Output:
        """Run Basic Pitch audio-to-MIDI transcription, then convert to MusicXML via music21."""
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

        out_dir = tempfile.mkdtemp()
        midi_path = os.path.join(out_dir, "output.mid")
        midi_data.write(midi_path)

        # Convert MIDI to MusicXML using music21
        musicxml_path = os.path.join(out_dir, "output.musicxml")
        midi_to_musicxml(midi_path, musicxml_path, stem_name)

        return Output(
            midi_file=Path(midi_path),
            musicxml_file=Path(musicxml_path),
        )


# ─── MIDI → MusicXML conversion via music21 ───────────────────────

# Standard guitar/bass tunings as MIDI note numbers
GUITAR_TUNING = [64, 59, 55, 50, 45, 40]  # E4 B3 G3 D3 A2 E2
BASS_TUNING = [55, 50, 45, 40]  # G2 D2 A1 E1

# MIDI drum map → General MIDI percussion names
DRUM_MAP = {
    35: "Acoustic Bass Drum", 36: "Bass Drum 1",
    37: "Side Stick", 38: "Acoustic Snare", 40: "Electric Snare",
    39: "Hand Clap",
    41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat", 45: "Low Tom", 46: "Open Hi-Hat",
    47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash Cymbal 1",
    50: "High Tom", 51: "Ride Cymbal 1", 52: "Chinese Cymbal",
    53: "Ride Bell", 54: "Tambourine", 55: "Splash Cymbal",
    56: "Cowbell", 57: "Crash Cymbal 2", 59: "Ride Cymbal 2",
}


def midi_to_musicxml(midi_path: str, output_path: str, stem_name: str = "") -> None:
    """Convert MIDI to MusicXML using music21 for high-quality notation."""
    from music21 import converter, instrument, stream, key, clef, meter, tempo
    from music21 import note as m21note, chord as m21chord, pitch as m21pitch
    from music21 import tablature

    stem = stem_name.lower().strip() if stem_name else ""

    # Parse MIDI with music21 (auto-quantizes)
    score = converter.parse(midi_path)

    # Flatten all parts into a single stream for processing
    flat = score.flatten().notesAndRests

    if len(flat) == 0:
        # Empty score — write minimal valid MusicXML
        s = stream.Score()
        p = stream.Part()
        m = stream.Measure()
        m.append(m21note.Rest(quarterLength=4.0))
        p.append(m)
        s.append(p)
        s.write("musicxml", fp=output_path)
        return

    # Determine notation mode
    is_tab = stem in ("guitar", "bass")
    is_drums = stem == "drums"
    is_bass_instrument = stem == "bass"

    # Create a new clean score
    new_score = stream.Score()
    part = stream.Part()

    # Set instrument
    if is_tab:
        if is_bass_instrument:
            inst = instrument.ElectricBass()
        else:
            inst = instrument.ElectricGuitar()
        part.insert(0, inst)
    elif is_drums:
        inst = instrument.Percussion()
        part.insert(0, inst)
    elif stem == "piano":
        inst = instrument.Piano()
        part.insert(0, inst)
    elif stem == "vocals":
        inst = instrument.Vocalist()
        part.insert(0, inst)
    else:
        # Auto-detect clef from pitch range
        pitches = [n.pitch.midi for n in flat if hasattr(n, "pitch")]
        if pitches:
            avg = sum(pitches) / len(pitches)
            if avg < 48:
                inst = instrument.ElectricBass()
            elif avg < 60:
                inst = instrument.Violoncello()  # Bass clef range
            else:
                inst = instrument.Piano()
        else:
            inst = instrument.Piano()
        part.insert(0, inst)

    # Analyze key signature (skip for drums)
    if not is_drums:
        try:
            detected_key = score.analyze("key")
            if detected_key:
                part.insert(0, detected_key)
        except Exception:
            pass

    # Get all measures from the original score, preserving quantization
    if score.parts:
        source_part = score.parts[0]
    else:
        # If no parts, make measures from the flat stream
        source_part = score.flatten()

    # music21 already quantized the MIDI — use makeMeasures for proper barlines
    if not source_part.getElementsByClass(stream.Measure):
        source_part = source_part.makeMeasures()

    # Copy measures into our new part
    for measure_obj in source_part.getElementsByClass(stream.Measure):
        new_measure = stream.Measure(number=measure_obj.number)

        # Copy time signature if present
        for ts in measure_obj.getElementsByClass(meter.TimeSignature):
            new_measure.insert(ts.offset, ts)

        # Copy tempo markings if present
        for mm in measure_obj.getElementsByClass(tempo.MetronomeMark):
            new_measure.insert(mm.offset, mm)

        # Copy notes and rests
        for element in measure_obj.notesAndRests:
            if is_drums and hasattr(element, "pitch"):
                # Convert pitched notes to unpitched percussion
                unpitched = m21note.Unpitched()
                unpitched.duration = element.duration
                midi_num = element.pitch.midi
                drum_name = DRUM_MAP.get(midi_num, f"Percussion {midi_num}")
                # Use display step/octave for percussion staff positioning
                if midi_num in (35, 36):  # Bass drum
                    unpitched.displayStep = "C"
                    unpitched.displayOctave = 4
                elif midi_num in (38, 40, 37):  # Snare
                    unpitched.displayStep = "E"
                    unpitched.displayOctave = 5
                elif midi_num in (42, 44):  # Closed hi-hat
                    unpitched.displayStep = "G"
                    unpitched.displayOctave = 5
                elif midi_num == 46:  # Open hi-hat
                    unpitched.displayStep = "A"
                    unpitched.displayOctave = 5
                elif midi_num in (49, 57):  # Crash
                    unpitched.displayStep = "A"
                    unpitched.displayOctave = 5
                elif midi_num in (51, 59, 53):  # Ride
                    unpitched.displayStep = "B"
                    unpitched.displayOctave = 5
                elif midi_num in (41, 43):  # Floor tom
                    unpitched.displayStep = "D"
                    unpitched.displayOctave = 4
                elif midi_num in (45, 47):  # Low-mid tom
                    unpitched.displayStep = "F"
                    unpitched.displayOctave = 4
                elif midi_num in (48, 50):  # High tom
                    unpitched.displayStep = "A"
                    unpitched.displayOctave = 4
                else:
                    unpitched.displayStep = "E"
                    unpitched.displayOctave = 4
                new_measure.insert(element.offset, unpitched)
            elif is_tab and hasattr(element, "pitch"):
                # Add string/fret technical notation for TAB
                tuning = BASS_TUNING if is_bass_instrument else GUITAR_TUNING
                string_num, fret_num = _midi_to_tab(element.pitch.midi, tuning)
                # Keep the note as-is; music21 will render TAB if instrument is set
                new_measure.insert(element.offset, element)
            else:
                new_measure.insert(element.offset, element)

        new_measure.makeBeams(inPlace=True)
        part.append(new_measure)

    new_score.append(part)

    # Set appropriate clef
    first_measure = part.getElementsByClass(stream.Measure).first()
    if first_measure:
        if is_drums:
            first_measure.insert(0, clef.PercussionClef())
        elif is_tab:
            first_measure.insert(0, clef.TabClef())

    # Write MusicXML
    new_score.write("musicxml", fp=output_path)


def _midi_to_tab(midi_note: int, tuning: list[int]) -> tuple[int, int]:
    """Find the best string and fret for a MIDI note on the given tuning."""
    best_string = 1
    best_fret = 0
    best_score = float("inf")
    for s, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= 24:
            score = fret + s * 0.1
            if score < best_score:
                best_score = score
                best_string = s + 1
                best_fret = fret
    if best_score == float("inf"):
        best_string = len(tuning)
        best_fret = max(0, midi_note - tuning[-1])
    return best_string, best_fret
