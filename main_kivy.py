################### Manensa4 ne7ot prediction 3la el person
import numpy as np
import joblib
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics import Color, RoundedRectangle
from kivy.properties import ListProperty


MODEL_PATH = r"Models\language_model_final_new.pkl"
ENCODER_PATH = "language_label_encoder_new.pkl"
WAV_PATH = "test_recording.wav"

SR = 22050
DURATION = 10  #seconds
N_MFCC = 17
N_CHROMA = 12
N_CONTRAST = 7
TOP_DB = 30

FLAG_PATHS = {
    "Arabic": r"pics\flag-of-egypt.jpg",
    "English": r"pics\Flag-United-Kingdom.webp",
    "French": r"pics\Flag_of_France.png",
    "German": r"pics\flag-of-germany.jpg",
}


# load model + encoder
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"Can not load model or encoder: {e}")
    model = None
    label_encoder = None


# Feature Extraction
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.normalize(mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=N_CONTRAST - 1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
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



# UI helpers
class RoundedButton(Button):
    bg_color = ListProperty([0.2, 0.6, 0.86, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (1, 1, 1, 1)
        self.bold = True
        with self.canvas.before:
            self._color = Color(*self.bg_color)
            self._rect = RoundedRectangle(radius=[16], pos=self.pos, size=self.size)
        self.bind(pos=self._update, size=self._update, bg_color=self._update_color)

    def _update(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def _update_color(self, *args):
        self._color.rgba = self.bg_color


class Card(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.13, 0.14, 0.17, 1)
            self._rect = RoundedRectangle(radius=[20], pos=self.pos, size=self.size)
        self.bind(pos=self._update, size=self._update)

    def _update(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size


# main
class VoiceDetector(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.padding = 25
        self.spacing = 15

        self.title = Label(text="Voice Language Detector",font_size=30,bold=True,color=(1, 1, 1, 1),size_hint=(1, 0.13))
        self.add_widget(self.title)
        self.status = Label(text="Press Record to start",font_size=18,color=(0.7, 0.75, 0.8, 1),size_hint=(1, 0.08))
        self.add_widget(self.status)

        self.record_btn = RoundedButton(text="Record (10s)",font_size=20,size_hint=(1, 0.11),bg_color=[0.85, 0.25, 0.35, 1])
        self.record_btn.bind(on_press=self.record)
        self.add_widget(self.record_btn)

        self.predict_btn = RoundedButton(text="Predict",font_size=20,size_hint=(1, 0.11),bg_color=[0.2, 0.6, 0.86, 1],disabled=True)
        self.predict_btn.bind(on_press=self.predict)
        self.add_widget(self.predict_btn)

        self.restart_btn = RoundedButton(text="Restart",font_size=20,size_hint=(1, 0.11),bg_color=[0.35, 0.38, 0.42, 1])
        self.restart_btn.bind(on_press=self.restart)
        self.add_widget(self.restart_btn)

        # Result card
        self.card = Card(orientation="vertical", padding=15, spacing=8, size_hint=(1, 0.46))

        self.flag = Image(source="", allow_stretch=True, size_hint=(1, 0.5))
        self.card.add_widget(self.flag)

        self.language = Label(text="Language: ", font_size=22, bold=True, color=(1, 1, 1, 1))
        self.card.add_widget(self.language)

        self.person = Label(text="Person: ", font_size=22, bold=True, color=(1, 1, 1, 1))
        self.card.add_widget(self.person)

        self.confidence = Label(text="Confidence: ", font_size=18, color=(0.7, 0.75, 0.8, 1))
        self.card.add_widget(self.confidence)

        self.add_widget(self.card)

        self._recording = None

    # Record 10s
    def record(self, instance):
        self.status.text = "Recording..."
        self.record_btn.disabled = True
        self.predict_btn.disabled = True

        # start async recording
        self._recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype=np.float32)

        # wait the duration then finish up
        Clock.schedule_once(self.record_finished, DURATION)

    def record_finished(self, dt):
        sd.wait()  #to ensure recording buffer is fully filled
        audio = self._recording.flatten()
        write(WAV_PATH, SR, audio)

        self.status.text = "Recording Complete"
        self.record_btn.disabled = False
        self.predict_btn.disabled = False

    # Predict
    def predict(self, instance):
        if model is None or label_encoder is None:
            self.status.text = "Model not loaded!"
            return

        self.status.text = "Predicting..."

        # preprocessing
        audio, _ = librosa.load(WAV_PATH, sr=SR)
        audio = librosa.effects.trim(audio, top_db=TOP_DB)[0]
        features = extract_features(audio, SR)
        features = features.reshape(1, -1)

        # prediction
        prediction = model.predict(features)
        language = label_encoder.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

        # person prediction
        #person = predict_person(features)
        person = "Unknown"
        
        # update UI
        self.language.text = f"Language: {language}"
        self.person.text = f"Person: {person}"
        self.confidence.text = f"Confidence: {confidence:.1f}%"
        self.flag.source = FLAG_PATHS.get(language, "")

        self.status.text = "Prediction Finished"


    # Restart (reset UI)
    def restart(self, instance):
        self.status.text = "Press Record to start"
        self.language.text = "Language: "
        self.person.text = "Person: "
        self.confidence.text = "Confidence: "
        self.flag.source = ""
        self.predict_btn.disabled = True
        self.record_btn.disabled = False


class MyApp(App):

    def build(self):
        Window.clearcolor = (0.07, 0.08, 0.1, 1)
        self.title = "Voice Language Detector"
        return VoiceDetector()


if __name__ == "__main__":
    MyApp().run()
