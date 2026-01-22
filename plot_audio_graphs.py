import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "dataset/sample"

def plot_audio(file_path, label):
    y, sr = librosa.load(file_path, sr=22050)

    # -------- Waveform --------
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {label}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # -------- Mel Spectrogram --------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {label}")
    plt.tight_layout()
    plt.show()


# -------- Loop through animal folders --------
for animal in os.listdir(DATASET_PATH):
    animal_path = os.path.join(DATASET_PATH, animal)

    if os.path.isdir(animal_path):
        for file in os.listdir(animal_path):
            if file.endswith(".wav"):
                file_path = os.path.join(animal_path, file)
                print(f"Processing: {animal}/{file}")
                plot_audio(file_path, animal)
