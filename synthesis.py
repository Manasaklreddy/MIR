from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import pretty_midi
import lameenc

def synthesize_from_f0(
    y: np.ndarray,
    sr: int,
    f0_hz: np.ndarray,
    hop_length: int = 512,
    rms: Optional[np.ndarray] = None,
    harmonics: int = 1,
) -> np.ndarray:
    """Additive sine synthesis from frame-wise f0 (Hz). Unvoiced frames (NaN) produce silence.

    Args:
        y: Reference audio (for duration only)
        sr: Sample rate
        f0_hz: Frame-wise f0 in Hz (NaN for unvoiced)
        hop_length: Hop length used to compute f0
        rms: Optional RMS per frame to use as amplitude envelope
        harmonics: Number of harmonics to add to the fundamental
    """
    n_samples = len(y)
    frame_count = len(f0_hz)
    # Create per-sample frequency by holding each frame value for hop_length samples
    freq_per_sample = np.repeat(f0_hz, hop_length)[:n_samples]
    freq_per_sample = np.nan_to_num(freq_per_sample, nan=0.0)

    # Amplitude envelope
    if rms is None:
        amp_frames = np.ones(frame_count)
    else:
        amp_frames = rms
    amp_per_sample = np.repeat(amp_frames, hop_length)[:n_samples]
    amp_per_sample = amp_per_sample / (np.max(amp_per_sample) + 1e-8)

    # Phase accumulator synthesis
    t = np.arange(n_samples) / sr
    phase = 2 * np.pi * np.cumsum(freq_per_sample) / sr
    out = np.zeros(n_samples, dtype=np.float32)
    if harmonics <= 1:
        out = (amp_per_sample * np.sin(phase)).astype(np.float32)
    else:
        acc = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, harmonics + 1):
            acc += (1.0 / h) * np.sin(h * phase)
        out = (amp_per_sample * acc / np.max(np.abs(acc) + 1e-8)).astype(np.float32)

    # Normalize
    out = out / (np.max(np.abs(out)) + 1e-8)
    return out.astype(np.float32)


def export_wav(audio: np.ndarray, sr: int, path: str) -> str:
    """Export audio array to WAV using soundfile."""
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def export_mp3(audio: np.ndarray, sr: int, path: str) -> str:
    """Export audio array to MP3 using lameenc (no FFmpeg required)."""
    # Normalize audio to int16
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(1 if audio.ndim == 1 else audio.shape[0])
    encoder.set_quality(2)  # 2=high, 7=fast

    mp3_data = encoder.encode(audio_int16.tobytes())
    mp3_data += encoder.flush()

    with open(path, "wb") as f:
        f.write(mp3_data)
    return path


def export_midi_from_f0(f0_hz: np.ndarray, sr: int, hop_length: int, path: str, instrument: str = "Acoustic Grand Piano") -> str:
    """Create a monophonic MIDI from frame-wise f0 by quantizing to nearest MIDI note.

    Unvoiced frames are treated as rests; contiguous voiced frames create a note.
    """
    pm = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument)
    inst = pretty_midi.Instrument(program=program)

    def hz_to_midi(hz: float) -> Optional[float]:
        return 69 + 12 * np.log2(hz / 440.0) if hz > 0 else None

    times = np.arange(len(f0_hz)) * (hop_length / sr)
    is_voiced = ~np.isnan(f0_hz)

    start_t = None
    note_midi = None
    for i, voiced in enumerate(is_voiced):
        if voiced:
            m = hz_to_midi(float(f0_hz[i]))
            if m is not None:
                if start_t is None:
                    start_t = times[i]
                    note_midi = int(np.round(m))
                elif int(np.round(m)) != note_midi:
                    # pitch change: end previous, start new
                    end_t = times[i]
                    inst.notes.append(pretty_midi.Note(velocity=80, pitch=note_midi, start=start_t, end=end_t))
                    start_t = times[i]
                    note_midi = int(np.round(m))
        else:
            if start_t is not None and note_midi is not None:
                end_t = times[i]
                inst.notes.append(pretty_midi.Note(velocity=80, pitch=note_midi, start=start_t, end=end_t))
                start_t = None
                note_midi = None

    # Close last note
    if start_t is not None and note_midi is not None:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=note_midi, start=start_t, end=times[-1] + (hop_length / sr)))

    pm.instruments.append(inst)
    pm.write(path)
    return path