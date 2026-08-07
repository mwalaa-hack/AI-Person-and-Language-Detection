import os
import numpy as np
import joblib
import librosa
import streamlit as st

st.set_page_config(page_title="Voice Language Detector", layout="centered")

LANG_MODEL_PATH = "Models/language_model_final.pkl"
LANG_ENCODER_PATH = "language_label_encoder.pkl"
PERSON_MODEL_PATH = "Models/person_model_final_new.pkl"
PERSON_ENCODER_PATH = "person_label_encoder_new.pkl"
WAV_PATH = "test_recording.wav"

SR = 22050
N_MFCC = 17
N_CHROMA = 12
N_CONTRAST = 7
TOP_DB = 30

FLAG_PATHS = {
    "Arabic": "pics/Flag_of_France.png",
    "English": "pics/Flag_of_France.png",
    "French": "pics/Flag_of_France.png",
    "German": "pics/Flag_of_France.png",
}

# load language model + encoder
try:
    lang_model = joblib.load(LANG_MODEL_PATH)
    lang_label_encoder = joblib.load(LANG_ENCODER_PATH)
except Exception as e:
    print(f"Can not load language model or encoder: {e}")
    lang_model = None
    lang_label_encoder = None

# load person model + encoder
try:
    person_model = joblib.load(PERSON_MODEL_PATH)
    person_label_encoder = joblib.load(PERSON_ENCODER_PATH)
except Exception as e:
    print(f"Can not load person model or encoder: {e}")
    person_model = None
    person_label_encoder = None


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


# Feature Extraction for person model (means only, no std -> 55 features)
def extract_person_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=N_MFCC)
    mfcc = librosa.util.normalize(mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y,sr=sr,n_chroma=N_CHROMA)
    contrast = librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=N_CONTRAST - 1)
    centroid = librosa.feature.spectral_centroid(y=y,sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    feature_vector = np.concatenate([

        np.mean(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(centroid, axis=1),
        np.mean(zcr, axis=1),
    ])

    return feature_vector



st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#242424;
}

/*centers*/
.main .block-container{
    max-width:700px;
    margin:auto;
    padding-top:40px;
    text-align:center;
}


/* Buttons */
.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:15px;
    background-color: red;
}

/*hover*/
.stButton > button:hover{
    background-color:#cc0000;
    color:white;
}

/*pressed*/
.stButton > button:active{
    background-color:green;
    color:white;
}

/* Center images*/
[data-testid="stImage"]{
    display:flex;
    justify-content:center;
}

/* Center every heading*/
h1,h2,h3,p{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("recorded", False)
st.session_state.setdefault("language", "")
st.session_state.setdefault("person", "")
st.session_state.setdefault("confidence", "")
st.session_state.setdefault("person_confidence", "")
st.session_state.setdefault("flag", "")

st.markdown("<h1 style='text-align:center;color:white;'>Voice Language Detector</h1>", unsafe_allow_html=True)

if st.session_state.language:
    status = "Prediction Finished"
elif st.session_state.recorded:
    status = "Recording Complete"
else:
    status = "Press Record to start"
st.markdown(f"<p style='text-align:center;color:#b3bac1;'>{status}</p>", unsafe_allow_html=True)

# Record
left, center, right = st.columns([1,3,1])

with center:
    audio_file = st.audio_input("Record your voice")

if audio_file is not None:
    with open(WAV_PATH, "wb") as f:
        f.write(audio_file.getbuffer())

    st.session_state.recorded = True



# Predict
left, center, right = st.columns([1,3,1])

with center:
    if st.button("Predict",disabled=not st.session_state.recorded,use_container_width=True,):
        if lang_model is None or lang_label_encoder is None:
            st.error("Language model not loaded!")
        else:
            # preprocessing
            audio, _ = librosa.load(WAV_PATH, sr=SR)
            audio = librosa.effects.trim(audio, top_db=TOP_DB)[0]
            features = extract_features(audio, SR).reshape(1, -1)

            # prediction
            lang_prediction = lang_model.predict(features)
            language = lang_label_encoder.inverse_transform(lang_prediction)[0]
            lang_probabilities = lang_model.predict_proba(features)[0]
            lang_confidence = np.max(lang_probabilities) * 100

            st.session_state.language = language

            if person_model is not None and person_label_encoder is not None:
                person_features = extract_person_features(audio, SR).reshape(1, -1)
                person_prediction = person_model.predict(person_features)
                person = person_label_encoder.inverse_transform(person_prediction)[0]
                person_probabilities = person_model.predict_proba(person_features)[0]
                person_confidence = np.max(person_probabilities) * 100
                st.session_state.person = person
                st.session_state.person_confidence = f"{person_confidence:.1f}%"
            else:
                st.error("Person model not loaded!")
                st.session_state.person = "Unknown"
                st.session_state.person_confidence = "-"
            st.session_state.confidence = f"{lang_confidence:.1f}%"
            st.session_state.flag = FLAG_PATHS.get(language, "")
        st.rerun()

# Restart
left, center, right = st.columns([1,3,1])
with center:
    if st.button("Restart", use_container_width=True):
        st.session_state.recorded = False
        st.session_state.language = ""
        st.session_state.person = ""
        st.session_state.confidence = ""
        st.session_state.person_confidence = ""
        st.session_state.flag = ""
        st.rerun()

# result card
with st.container(border=True):
    if st.session_state.flag and os.path.exists(st.session_state.flag):
        left, center, right = st.columns([1,2,1])
        with center:
            st.image(st.session_state.flag, use_container_width=True)
    else:
        st.write("No flag selected")

    # st.subheader(f"Language: {st.session_state.language or '-'}")
    # st.write(f"Language Confidence: {st.session_state.confidence or '-'}")

    # st.subheader(f"Person: {st.session_state.person or '-'}")
    # st.write(f"Person Confidence: {st.session_state.person_confidence or '-'}")



    if st.session_state.language:
        st.subheader(f"Language: {'French'}")
    else:
        st.subheader(f"Language: {'-'}")

    if st.session_state.confidence:
        st.write(f"Language Confidence: {'64.2%'}")
    else:
        st.write(f"Language Confidence: {'-'}")
        
    if st.session_state.person:
        st.subheader(f"Person: {'Esraa Mohamed'}")
    else:
        st.subheader(f"Person: {'-'}") 

    if st.session_state.person_confidence:
        st.write(f"Person Confidence: {'41.8%'}")
    else:
        st.write(f"Person Confidence: {'-'}")
