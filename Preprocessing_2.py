import librosa
import numpy as np
import os
import pandas as pd

data_path = "./Dataset"
SR = 22050           # sample rate to resample every file to
N_MFCC = 17           # number of MFCC coefficients
N_CHROMA = 12         # number of chroma bins
N_CONTRAST = 7        # number of spectral contrast bands
TOP_DB = 30           # silence threshold for trimming


def extract_features(y, sr):
 
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

    # Collapse every feature's time axis to its mean, then concatenate into one vector
        # Use both mean and standard deviation for each feature over time
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


Person = []
Language = []
Features = []

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    if not os.path.isdir(person_path):
        continue

    for language in os.listdir(person_path):
        language_path = os.path.join(person_path, language)
        if not os.path.isdir(language_path):
            continue

        for voice in os.listdir(language_path):
            if not voice.endswith('.wav'):
                continue

            audio_path = os.path.join(language_path, voice)

            librosa_audio, sr = librosa.load(audio_path, sr=SR)              # load audio file
            removed_silence = librosa.effects.trim(librosa_audio, top_db=TOP_DB)[0]  # remove silence

            feature_vector = extract_features(removed_silence, SR)

            Person.append(person)
            Language.append(language)
            Features.append(feature_vector)

Features = np.array(Features)
Person = np.array(Person)
Language = np.array(Language)

print("Features shape:", Features.shape)
print("Person shape:", Person.shape)
print("Language shape:", Language.shape)
print("Languages found:", np.unique(Language))
print("Persons found:", np.unique(Person))

feature_names = (
    [f"mfcc_mean_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_std_{i+1}" for i in range(N_MFCC)] +

    [f"mfcc_delta_mean_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_delta_std_{i+1}" for i in range(N_MFCC)] +

    [f"chroma_mean_{i+1}" for i in range(N_CHROMA)] +
    [f"chroma_std_{i+1}" for i in range(N_CHROMA)] +

    [f"contrast_mean_{i+1}" for i in range(N_CONTRAST)] +
    [f"contrast_std_{i+1}" for i in range(N_CONTRAST)] +

    ["spectral_centroid_mean", "spectral_centroid_std"] +
    ["zero_crossing_rate_mean", "zero_crossing_rate_std"]
)

df = pd.DataFrame(Features, columns=feature_names)
df["Person"] = Person
df["Language"] = Language

df.to_csv("preprocessed_features_2.csv", index=False)
print("Saved preprocessed_features_2.csv with shape:", df.shape)
