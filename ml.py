from typing import List, Tuple, Dict
import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import librosa


GENRES = [
	"blues", "classical", "country", "disco", "hiphop",
	"jazz", "metal", "pop", "reggae", "rock"
]


def extract_file_features(path: str, sr: int = 22050, hop_length: int = 512, n_mfcc: int = 13) -> np.ndarray:
	y, sr = librosa.load(path, sr=sr, mono=True)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
	spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
	zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
	rms = librosa.feature.rms(y=y, hop_length=hop_length)
	# summarize per track
	def summarize(x: np.ndarray) -> List[float]:
		return [float(np.nanmean(x)), float(np.nanstd(x)), float(np.nanmin(x)), float(np.nanmax(x))]
	features = []
	for i in range(n_mfcc):
		features += summarize(mfcc[i])
	features += summarize(spectral_centroid)
	features += summarize(zcr)
	features += summarize(rms)
	return np.array(features, dtype=np.float32)


def build_dataset(gtzan_root: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Build feature matrix X and label vector y from GTZAN directory structure."""
	X, y = [], []
	for genre in GENRES:
		pattern = os.path.join(gtzan_root, genre, "*.wav")
		for path in glob.glob(pattern):
			X.append(extract_file_features(path))
			y.append(GENRES.index(genre))
	return np.vstack(X), np.array(y)


def train_and_evaluate(X: np.ndarray, y: np.ndarray, model: str = "svm") -> Dict[str, any]:
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
	if model == "svm":
		clf = Pipeline([
			("scaler", StandardScaler()),
			("svc", SVC(kernel="rbf", C=10, gamma="scale"))
		])
	else:
		clf = RandomForestClassifier(n_estimators=300, random_state=42)

	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, target_names=GENRES)
	cm = confusion_matrix(y_test, y_pred)
	return {"model": clf, "accuracy": acc, "report": report, "confusion_matrix": cm}


def save_model(model, path: str) -> str:
	joblib.dump(model, path)
	return path


def load_model(path: str):
	return joblib.load(path) 