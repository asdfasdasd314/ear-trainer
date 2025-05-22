import audio
import librosa
import numpy as np
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from constants import RATE, CHUNK
import plot
import soundfile as sf
import data
from torch.utils.data import DataLoader
import ml
import torch

def main():
    # microphone_audio = audio.collect_microphone_audio()
    # audio.save_audio(microphone_audio, "microphone_audio.wav") # To load with essentia
    pitches, _, _ = audio.extract_melody("musdb18hq/train/A Classic Education - NightOwl/mixture.wav")
    y, _ = librosa.load("musdb18hq/train/A Classic Education - NightOwl/mixture.wav", sr=RATE)
    harmonic = librosa.effects.hpss(y, n_fft=CHUNK, hop_length=CHUNK // 2, margin=(1.0, 5.0), power=2.0)[0]
    S = librosa.stft(harmonic, n_fft=CHUNK, hop_length=CHUNK // 2, center=True)
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

    # Ensure both are the same length
    min_len = min(S.shape[1], len(pitches))
    S_db = S_db[:, :min_len]
    pitches = pitches[:min_len]

    train_dataset = data.MelodyHarmonyDataset("musdb18hq/train")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    test_dataset = data.MelodyHarmonyDataset("musdb18hq/test")
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    model = ml.LSTM(hidden_size=100, num_layers=1)
    ml.train_model(model, train_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.01), 100)
    ml.save_model(model)
    ml.test_model(model, test_loader)

if __name__ == "__main__":
    main()