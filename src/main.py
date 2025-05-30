from constants import CHUNK
import data
import plot
import ml
import torch
import audio
import librosa
import numpy as np
from torch.utils.data import DataLoader

def main():
    # Load a song
    y = audio.load_song("musdb18hq/train/A Classic Education - NightOwl")
    harmonic, _ = librosa.effects.hpss(y)
    S = librosa.stft(harmonic, n_fft=CHUNK)
    S = np.abs(S).T # Transpose it
    plot.plot_spectrogram(S)
    

if __name__ == "__main__":
    main()