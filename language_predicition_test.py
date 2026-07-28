import joblib
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

MODEL_PATH = r"Models\language_model_final.pkl"
ENCODER_PATH = "language_label_encoder.pkl"

SR = 22050
DURATION = 10 #seconds

N_MFCC = 17
N_CHROMA = 12
N_CONTRAST = 7
TOP_DB = 30


# Load model
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# Feature Extraction
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)
    mfcc = librosa.util.normalize(mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y,sr=sr,n_chroma=N_CHROMA)
    contrast = librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=N_CONTRAST - 1)
    centroid = librosa.feature.spectral_centroid(y=y,sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    feature_vector = np.concatenate([

        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),

        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),

        np.mean(contrast, axis=1),
        np.std(contrast, axis=1),

        np.mean(centroid, axis=1),
        np.std(centroid, axis=1),

        np.mean(zcr, axis=1),
        np.std(zcr, axis=1),
    ])

    return feature_vector


# Record Audio
print("Language Prediction Test")
input("\nPress ENTER to start recording...")
print("Recording...")
audio = sd.rec(int(DURATION * SR),samplerate=SR,channels=1,dtype=np.float32)
sd.wait()
print("Recording finished.\n")
audio = audio.flatten()
write("test_recording.wav", SR, audio)


# Preprocess
audio, _ = librosa.load("test_recording.wav",sr=SR)
audio = librosa.effects.trim(audio,top_db=TOP_DB)[0]
features = extract_features(audio, SR)
features = features.reshape(1, -1)


# Predict
prediction = model.predict(features)
language = label_encoder.inverse_transform(prediction)[0]
probabilities = model.predict_proba(features)[0]

print("Prediction")
print(f"\nPredicted Language : {language}")
print("\nProbabilities:")
for lang, prob in zip(label_encoder.classes_, probabilities):
    print(f"{lang:<10}: {prob*100:.2f}%")