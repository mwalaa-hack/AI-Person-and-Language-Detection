import joblib
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

MODEL_PATH = "person_model_final_new.pkl"
ENCODER_PATH = "person_label_encoder_new.pkl"

DURATION = 5  # seconds
SR = 22050            # sample rate to resample every file to
N_MFCC = 17            # number of MFCC coefficients
N_CHROMA = 12          # number of chroma bins
N_CONTRAST = 7         # number of spectral contrast bands
TOP_DB = 30            # silence threshold for trimming

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load model
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# Feature Extraction (MUST match training exactly - mean only, same order)
def extract_features(y, sr):
    """Build one fixed-length feature vector for a single (already-trimmed) audio signal."""

    # MFCCs + deltas: phonetic/timbre content and how it changes over time
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.normalize(mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Chroma: pitch-class / tonal content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)

    # Spectral contrast: peak-vs-valley energy, useful speaker cue
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=N_CONTRAST - 1)

    # Spectral centroid: brightness of the voice
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Zero-crossing rate: noisiness / consonant vs. vowel balance
    zcr = librosa.feature.zero_crossing_rate(y=y)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(centroid, axis=1),
        np.mean(zcr, axis=1),
    ])




# Record Audio
print("Person Identification Test")
input("\nPress ENTER to start recording...")
print("Recording...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype=np.float32)
sd.wait()
print("Recording finished.\n")
audio = audio.flatten()
write("test_recording.wav", SR, audio)


# Preprocess
audio, _ = librosa.load("test_recording.wav", sr=SR)
audio = librosa.effects.trim(audio, top_db=TOP_DB)[0]
features = extract_features(audio, SR)
features = features.reshape(1, -1)


# Predict
prediction = model.predict(features)
person = label_encoder.inverse_transform(prediction)[0]
probabilities = model.predict_proba(features)[0]

print("Prediction")
print(f"\nPredicted Person : {person}")
print("\nProbabilities:")
for name, prob in zip(label_encoder.classes_, probabilities):
    print(f"{name:<10}: {prob*100:.2f}%")


'''
Prediction

Predicted Person : MariamB

Probabilities:
EsraaM    : 23.22%
MWalaa    : 9.55%
MariamB   : 67.23%
'''