import gradio as gr
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io

# Parametry
SR = 22050
N_MELS = 128
TARGET_FRAMES = 216
LABELS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
polskie_nazwy = {
    'cel': 'wiolonczela',
    'cla': 'klawesyn',
    'flu': 'flet',
    'gac': 'gitara klasyczna',
    'gel': 'gitara elektryczna',
    'org': 'organy',
    'pia': 'fortepian',
    'sax': 'saksofon',
    'tru': 'trąbka',
    'vio': 'skrzypce',
    'voi': 'głos ludzki'
}
dark_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="gray",
    font="sans"
).set(
    body_background_fill="#121212",
    block_background_fill="#1E1E1E",
    body_text_color="#ffffff",
    button_primary_background_fill="#2d72d9",
    button_primary_text_color="#ffffff"
)

# Wczytanie modelu
model = tf.keras.models.load_model("model.h5")

def compute_melspectrogram(y, sr=SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def resize_spectrogram(S, target_frames=TARGET_FRAMES):
    if S.shape[1] < target_frames:
        pad = target_frames - S.shape[1]
        left = pad // 2
        right = pad - left
        S = np.pad(S, ((0, 0), (left, right)), mode='constant')
    elif S.shape[1] > target_frames:
        start = (S.shape[1] - target_frames) // 2
        S = S[:, start:start+target_frames]
    return S

def predict_and_plot(audio_path):
    y, _ = librosa.load(audio_path, sr=SR)
    
    S_full = compute_melspectrogram(y)
    S = resize_spectrogram(S_full)
    
    x = S[np.newaxis, ..., np.newaxis]
    preds = model.predict(x, verbose=0)[0]
    

    fig, ax = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(S_full, sr=SR, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set_title("Mel-spektrogram")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)

    pred_dict = {polskie_nazwy[label]: float(p) for label, p in zip(LABELS, preds)}
    return pred_dict, image

demo = gr.Interface(
    fn=predict_and_plot,
    inputs=gr.Audio(type="filepath", label="Wgraj plik WAV"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predykcja"),
        gr.Image(label="Spektrogram")
    ],
    title="Rozpoznawanie instrumentów",
    description="Model klasyfikuje dźwięki do kilku z klas instrumentów.",
    theme=dark_theme,
    submit_btn="Zatwierdź",
    clear_btn="Wyczyść"
)

if __name__ == "__main__":
    demo.launch()