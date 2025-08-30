import io
import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import librosa

from feature_extraction import extract_features, save_features_csv
from analysis import compute_stft, detect_pitch_pyinn, compute_chromagram
from visualization import plot_waveform, plot_spectrogram, plot_chromagram, plot_beats
from synthesis import synthesize_from_f0, export_wav, export_mp3, export_midi_from_f0
from separation import separate_stems, save_stems

st.set_page_config(page_title="MIR Toolkit", layout="wide")
st.title("Music Information Retrieval (MIR) Toolkit")

# Sidebar - configuration
sr = st.sidebar.number_input("Sample rate", min_value=8000, max_value=48000, value=22050, step=1000)
hop_length = st.sidebar.number_input("Hop length", min_value=128, max_value=4096, value=512, step=128)

# Separation options
st.sidebar.header("Audio Separation")
separation_enabled = st.sidebar.checkbox("Enable Audio Separation", value=False)
stems_count = st.sidebar.selectbox("Number of stems", [2, 4], index=1, 
                                  help="2: vocals/accompaniment, 4: vocals/drums/bass/other")

uploaded = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3", "flac", "ogg", "m4a"]) 

if uploaded is not None:
	st.subheader("Original Audio")
	data = uploaded.read()
	with open("_tmp_input", "wb") as f:
		f.write(data)
	y, sr_loaded = librosa.load("_tmp_input", sr=sr, mono=True)
	st.audio(data, format=f"audio/{uploaded.type.split('/')[-1] if '/' in uploaded.type else 'wav'}")

	# Audio Separation
	if separation_enabled:
		st.header("1. Audio Separation")
		with st.spinner("Separating audio into stems..."):
			try:
				stems = separate_stems(y, sr, stems=stems_count)
				stem_paths = save_stems(stems, sr, "_tmp_stems")
				st.success(f"Successfully separated into {len(stems)} stems!")
				
				# Display stems
				col1, col2 = st.columns(2)
				with col1:
					st.subheader("Stem Audio")
					for name, audio in stems.items():
						st.write(f"**{name.title()}**")
						st.audio(audio, sample_rate=sr)
						
				with col2:
					st.subheader("Stem Waveforms")
					for name, audio in stems.items():
						st.write(f"**{name.title()}:**")
						st.pyplot(plot_waveform(audio, sr))
						
				# Stem analysis
				st.subheader("Stem Analysis")
				stem_tab1, stem_tab2, stem_tab3 = st.tabs(["Features", "Spectrograms", "Download"])
				
				with stem_tab1:
					stem_features = {}
					for name, audio in stems.items():
						features = extract_features(audio, sr=sr, hop_length=hop_length, n_mfcc=13)
						stem_features[name] = features
						st.write(f"**{name.title()} Features:**")
						st.json({
							"tempo": features["tempo"],
							"beats": len(features["beat_times"]),
							"rms_mean": float(np.mean(features["rms"])),
						})
				
				with stem_tab2:
					for name, audio in stems.items():
						st.write(f"**{name.title()} Spectrogram:**")
						stft = compute_stft(audio, sr=sr, hop_length=hop_length)
						st.pyplot(plot_spectrogram(stft.get("S_db", stft["S"]), sr=sr, hop_length=hop_length))
				
				with stem_tab3:
					col1, col2 = st.columns(2)
					with col1:
						# Download individual stems
						for name, audio in stems.items():
							wav_bytes = io.BytesIO()
							wav_path = export_wav(audio, sr, f"_tmp_stems/{name}.wav")
							with open(wav_path, "rb") as f:
								wav_bytes.write(f.read())
							st.download_button(f"Download {name.title()} WAV", 
											data=wav_bytes.getvalue(), 
											file_name=f"{name}.wav", 
											mime="audio/wav")
					
					with col2:
						# Download all stems as ZIP
						st.info("Individual stems are saved in the _tmp_stems/ folder")
						st.write("Available stems:")
						for name in stems.keys():
							st.write(f"• {name.title()}")
				
			except Exception as e:
				st.error(f"Separation failed: {e}")
				st.info("Make sure you have the required dependencies installed (torch, demucs)")

	# Feature Extraction
	st.header("2. Feature Extraction")
	features = extract_features(y, sr=sr, hop_length=hop_length, n_mfcc=13)
	st.json({
		"tempo": features["tempo"],
		"beats": len(features["beat_times"]),
		"rms_mean": float(np.mean(features["rms"])),
	})
	
	col1, col2 = st.columns(2)
	with col1:
		st.pyplot(plot_waveform(y, sr))
	with col2:
		st.pyplot(plot_beats(y, sr, features["beat_times"]))

	# Core Analysis
	st.header("3. Core Analysis")
	stft = compute_stft(y, sr=sr, hop_length=hop_length)
	st.pyplot(plot_spectrogram(stft.get("S_db", stft["S"]), sr=sr, hop_length=hop_length))

	f0, voiced_prob = detect_pitch_pyinn(y, sr=sr, hop_length=hop_length)
	st.write(f"Pitch frames: {np.sum(~np.isnan(f0))} voiced / {len(f0)} total")

	chroma = compute_chromagram(y, sr=sr, hop_length=hop_length)
	st.pyplot(plot_chromagram(chroma, sr=sr, hop_length=hop_length))

	# Download features CSV
	csv_df = pd.DataFrame()
	try:
		csv_df = pd.DataFrame({"frame": np.arange(features["mfcc"].shape[1])})
		for i in range(features["mfcc"].shape[0]):
			csv_df[f"mfcc_{i+1}"] = features["mfcc"][i]
		csv_df["spectral_centroid"] = features["spectral_centroid"][0]
		csv_df["zcr"] = features["zcr"][0]
		csv_df["rms"] = features["rms"][0]
		csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
		st.download_button("Download Features CSV", data=csv_bytes, file_name="features.csv", mime="text/csv")
	except Exception as e:
		st.warning(f"Could not prepare CSV: {e}")

	# Synthesis
	st.header("4. Audio Synthesis")
	harmonics = st.slider("Harmonics", 1, 8, 1, 1)
	use_rms = st.checkbox("Use RMS as amplitude envelope", value=True)
	rms_frames = features["rms"][0] if use_rms else None
	with st.spinner("Synthesizing from f0 ..."):
		y_synth = synthesize_from_f0(y, sr, f0, hop_length=hop_length, rms=rms_frames, harmonics=harmonics)
	st.audio(y_synth, sample_rate=sr)
	colw1, colw2, colw3 = st.columns(3)
	with colw1:
		wav_bytes = io.BytesIO()
		wav_path = export_wav(y_synth, sr, "_tmp_synth.wav")
		with open(wav_path, "rb") as f:
			wav_bytes.write(f.read())
		st.download_button("Download Synth WAV", data=wav_bytes.getvalue(), file_name="synth.wav", mime="audio/wav")
	with colw2:
		try:
			mp3_path = export_mp3(y_synth, sr, "_tmp_synth.mp3")
			with open(mp3_path, "rb") as f:
				st.download_button("Download Synth MP3", data=f.read(), file_name="synth.mp3", mime="audio/mpeg")
		except Exception as e:
			st.info(f"MP3 export unavailable: {e}")
	with colw3:
		midi_path = export_midi_from_f0(f0, sr, hop_length, "_tmp_synth.mid")
		with open(midi_path, "rb") as f:
			st.download_button("Download MIDI", data=f.read(), file_name="synth.mid", mime="audio/midi")

	st.caption("Note: For large files, operations may take time. Ensure FFmpeg is installed for MP3 export.")
else:
	st.info("Upload an audio file to begin.") 