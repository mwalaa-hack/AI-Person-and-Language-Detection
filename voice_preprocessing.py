"""
Voice Preprocessing: Person + Language Classification
------------------------------------------------------
Prepares features for a model that predicts WHO is speaking and WHICH
language (Arabic, English, German, French) they are speaking.

Leakage-safe design:
  1. Collect all file paths first (no audio processing).
  2. Split file paths into train/test BEFORE any augmentation.
  3. Augment ONLY the training files (several augmented copies per file).
  4. Test files are processed as originals only — never augmented.
  5. Train and test features are saved to two SEPARATE CSV files.

Assumed folder structure (files already .wav, no format conversion needed):
    Data/
      person_1/
        Arabic/*.wav
        English/*.wav
        German/*.wav
        French/*.wav
      person_2/
        ...
If your folders are structured differently, adjust collect_files() below.
"""

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "./Dataset"

SR = 22050            # sample rate to resample every file to
N_MFCC = 17            # number of MFCC coefficients
N_CHROMA = 12          # number of chroma bins
N_CONTRAST = 7         # number of spectral contrast bands
TOP_DB = 30            # silence threshold for trimming

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Augmentations applied ONLY to the training split.
AUGMENTATIONS = ["noise", "time_stretch", "pitch_shift", "time_shift", "volume"]
N_AUGMENTED_COPIES = 3  # augmented versions generated per training recording

TRAIN_OUTPUT_CSV = "train_features.csv"
TEST_OUTPUT_CSV = "test_features.csv"


# ---------------------------------------------------------------------------
# Step 1 — Collect file paths (no audio processing yet)
# ---------------------------------------------------------------------------
def collect_files(data_path):
    records = []
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path):
            continue

        for language in os.listdir(person_path):
            language_path = os.path.join(person_path, language)
            if not os.path.isdir(language_path):
                continue

            for voice in os.listdir(language_path):
                if not voice.endswith(".wav"):
                    continue

                records.append({
                    "path": os.path.join(language_path, voice),
                    "person": person,
                    "language": language,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 2 — Augmentation functions (train split only)
# ---------------------------------------------------------------------------
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def time_stretch(y, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)  # 15% slower to 15% faster
    return librosa.effects.time_stretch(y=y, rate=rate)


def pitch_shift(y, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)  # +/- 2 semitones
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_shift(y, shift_max=0.2):
    shift_amount = int(len(y) * np.random.uniform(-shift_max, shift_max))
    return np.roll(y, shift_amount)


def change_volume(y, gain_range=(0.6, 1.4)):
    gain = np.random.uniform(*gain_range)
    return y * gain


def augment_audio(y, sr, augmentation_name):
    if augmentation_name == "noise":
        return add_noise(y)
    elif augmentation_name == "time_stretch":
        return time_stretch(y)
    elif augmentation_name == "pitch_shift":
        return pitch_shift(y, sr)
    elif augmentation_name == "time_shift":
        return time_shift(y)
    elif augmentation_name == "volume":
        return change_volume(y)
    else:
        raise ValueError(f"Unknown augmentation: {augmentation_name}")


# ---------------------------------------------------------------------------
# Step 3 — Feature extraction
# ---------------------------------------------------------------------------
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


def load_and_trim(path):
    audio, sr = librosa.load(path, sr=SR)
    return librosa.effects.trim(audio, top_db=TOP_DB)[0]


def process_split(rows, augment):
    """Process a set of file rows into feature vectors.

    If augment=True (training split only), also generates N_AUGMENTED_COPIES
    augmented versions per file. If augment=False (test split), only the
    original recording's features are extracted.
    """
    persons, languages, augmentations, features = [], [], [], []

    for _, row in rows.iterrows():
        removed_silence = load_and_trim(row["path"])

        # original recording
        features.append(extract_features(removed_silence, SR))
        persons.append(row["person"])
        languages.append(row["language"])
        augmentations.append("original")

        if augment:
            for _ in range(N_AUGMENTED_COPIES):
                augmentation_name = np.random.choice(AUGMENTATIONS)
                try:
                    augmented_audio = augment_audio(removed_silence, SR, augmentation_name)
                except Exception as e:
                    print(f"Skipped augmentation '{augmentation_name}' on {row['path']}: {e}")
                    continue

                features.append(extract_features(augmented_audio, SR))
                persons.append(row["person"])
                languages.append(row["language"])
                augmentations.append(augmentation_name)

    feature_names = (
        [f"mfcc_{i+1}" for i in range(N_MFCC)]
        + [f"mfcc_delta_{i+1}" for i in range(N_MFCC)]
        + [f"chroma_{i+1}" for i in range(N_CHROMA)]
        + [f"contrast_{i+1}" for i in range(N_CONTRAST)]
        + ["spectral_centroid"]
        + ["zero_crossing_rate"]
    )

    df = pd.DataFrame(features, columns=feature_names)
    df["Person"] = persons
    df["Language"] = languages
    df["Augmentation"] = augmentations
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Collect files (no audio touched yet)
    files_df = collect_files(DATA_PATH)
    print("Total files found:", len(files_df))
    print("Persons found:", sorted(files_df["person"].unique()))
    print("Languages found:", sorted(files_df["language"].unique()))

    # 2. Split BEFORE augmentation, stratified by person+language
    combined_label = files_df["person"] + "_" + files_df["language"]
    try:
        train_files, test_files = train_test_split(
            files_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=combined_label
        )
    except ValueError:
        print("Not enough samples per person+language combo to stratify — using a random split instead.")
        train_files, test_files = train_test_split(
            files_df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    print("Train files:", len(train_files))
    print("Test files:", len(test_files))

    # 3. Process each split: train gets augmented, test does not
    train_df = process_split(train_files, augment=True)
    test_df = process_split(test_files, augment=False)

    print("Train rows (original + augmented):", len(train_df))
    print("Test rows (original only):", len(test_df))
    print("Augmentation counts (train only):", train_df["Augmentation"].value_counts().to_dict())

    # 4. Save to two SEPARATE files
    train_df.to_csv(TRAIN_OUTPUT_CSV, index=False)
    test_df.to_csv(TEST_OUTPUT_CSV, index=False)
    print(f"Saved {TRAIN_OUTPUT_CSV} with shape {train_df.shape}")
    print(f"Saved {TEST_OUTPUT_CSV} with shape {test_df.shape}")


if __name__ == "__main__":
    main()
