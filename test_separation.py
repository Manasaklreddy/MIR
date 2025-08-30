from separation import separate_stems, save_stems
import librosa

y, sr = librosa.load("audio.wav", sr=44100, mono=True)
stems = separate_stems(y, sr, stems=4)
save_stems(stems, sr, "_test_stems")