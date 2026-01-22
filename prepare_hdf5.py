import os
import librosa
import h5py
import numpy as np

DATASET_PATH = "dataset/sample"
OUTPUT_H5 = "dataset/animal_sample.h5"

sr = 32000
max_len = sr * 10
n_mels = 64

X, y = [], []
label_map = {}
label_id = 0

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    label_map[label] = label_id

    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(label_path, file)
            audio, _ = librosa.load(file_path, sr=sr)

            if len(audio) > max_len:
                audio = audio[:max_len]
            else:
                audio = np.pad(audio, (0, max_len - len(audio)))

            mel = librosa.feature.melspectrogram(
                audio, sr=sr, n_mels=n_mels
            )
            log_mel = librosa.power_to_db(mel)

            X.append(log_mel)
            y.append(label_id)

    label_id += 1

X = np.array(X)
y = np.array(y)

with h5py.File(OUTPUT_H5, "w") as hf:
    hf.create_dataset("features", data=X)
    hf.create_dataset("labels", data=y)

print("HDF5 created at:", OUTPUT_H5)
print("Label mapping:", label_map)
