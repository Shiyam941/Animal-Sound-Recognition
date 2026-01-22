🐾 Animal Sound Recognition Using Deep Learning

This project performs animal sound recognition using deep learning techniques.
It extracts audio features (Mel Spectrograms) from .wav files and stores them in HDF5 format, which can later be used for training CNN-based models.

Supported example classes:
Dog
Cat
Hen
Pigeon

📁 Project Structure
Animal-sound-recognition-main/
│
├── dataset/
│   ├── sample/
│   │   ├── dog/
│   │   ├── cat/
│   │   ├── hen/
│   │   └── pigeon/
│   └── animal_sample.h5
│
├── prepare_hdf5.py
├── test_hdf5.py
├── plot_audio_graphs.py
├── requirements.txt
├── venv/
└── README.md

💻 System Requirements
Requirement	Version
OS	Windows 10 / 11
Python	Python 3.8.x only
RAM	8 GB or more (recommended)
Internet	Required for installing packages

⚠️ Important
Do NOT use Python 3.9 / 3.10 / 3.11 / 3.12 / 3.14
This project works correctly ONLY with Python 3.8

🧩 Step 1: Install Python (NEW LAPTOP)

Download Python 3.8.10 from:

https://www.python.org/downloads/release/python-3810/


During installation:

✅ Check Add Python to PATH

✅ Select pip

Verify installation:

python --version


Output should be:

Python 3.8.10

🧪 Step 2: Create Virtual Environment

Open Command Prompt inside the project folder:

python -m venv venv
venv\Scripts\activate


You should see:

(venv)

📦 Step 3: Install Required Libraries (IMPORTANT)

Run this single command (tested and stable):

python -m pip install ^
numpy==1.19.5 ^
numba==0.48.0 ^
llvmlite==0.31.0 ^
scipy==1.5.4 ^
scikit-learn==0.24.2 ^
joblib==1.1.1 ^
soundfile==0.10.3.post1 ^
audioread==2.1.9 ^
decorator==4.4.2 ^
resampy==0.2.2 ^
librosa==0.8.1 ^
h5py==2.10.0 ^
torch==1.10.2 ^
torchlibrosa==0.1.0 ^
matplotlib==3.3.4 ^
--trusted-host pypi.org --trusted-host files.pythonhosted.org


✅ These versions avoid all NumPy / Librosa / Numba conflicts.

🔍 Step 4: Verify Installation
python -c "import numpy, librosa, numba, resampy, h5py, matplotlib; print('All OK')"


If no error appears → ✅ Environment ready

🎧 Step 5: Prepare Dataset

Place your .wav files like this:

dataset/sample/dog/dog1.wav
dataset/sample/cat/cat1.wav
dataset/sample/hen/hen1.wav
dataset/sample/pigeon/pigeon1.wav


Each folder name = class label

🧱 Step 6: Create HDF5 Dataset

Run:

python prepare_hdf5.py


Expected output:

HDF5 created at: dataset/animal_sample.h5
Label mapping: {'cat': 0, 'dog': 1, 'hen': 2, 'pigeon': 3}

✅ Step 7: Verify HDF5 File
python test_hdf5.py


Example output:

Features shape: (4, 64, 626)
Labels: [0 1 2 3]

📊 Step 8: Plot Graphs for Each Audio

Run:

python plot_audio_graphs.py


This will display:

Waveform graph

Mel Spectrogram

For each animal sound

Close one graph window to view the next.

🧠 What This Project Does (Simple Explanation)

Reads .wav audio files

Converts audio into Mel Spectrogram features

Stores features + labels in HDF5 format

Visualizes sound patterns using graphs

Prepares data for deep learning models (CNNs)

🚫 Common Mistakes to Avoid

❌ Using Python > 3.8
❌ Upgrading NumPy blindly
❌ Installing latest matplotlib / librosa
❌ Running without activating venv

🚀 Future Extensions

Train CNN / ResNet model

Add more animal classes

Real-time microphone prediction

Mobile app integration

Accuracy / loss graphs

📌 Author Note

This project was executed using carefully selected library versions to maintain compatibility with legacy deep learning audio frameworks.