from typing import List, Tuple
import numpy as np
import librosa
import essentia.standard as es
import wave
import pyaudio
import threading
import os
import music21
import torch
from transformer import ChordTransformer, max_len
from constants import CHROMA_TO_NOTE, RATE, CHUNK, FORMAT, CHANNELS

def mask_noise(spectra, threshold):
    pass


def load_song(song_path: str) -> np.ndarray:
    y, _ = librosa.load(os.path.join(song_path, "mixture.wav"), sr=RATE)
    return y


def compute_weighted_chroma(spectrum: np.ndarray, chroma: np.ndarray) -> Tuple[np.ndarray, float]:
    weighted_chroma = np.zeros(12)
    for i in range(len(chroma)):
        chroma_idx = int(chroma[i])
        weighted_chroma[chroma_idx] += spectrum[i]
    if np.sum(weighted_chroma) == 0:
        return np.zeros(12)
    return np.square(weighted_chroma / np.sum(weighted_chroma))


def construct_chord_symbol(root: int, midi: List[int]) -> str:
    chord = music21.chord.Chord(midi)
    chord.root = music21.pitch.Pitch(midi=root)
    symbol = music21.harmony.ChordSymbol()
    for note in chord.pitches:
        symbol.add(note)
    return symbol.figure


def chroma_to_midi(chroma: List[int]) -> List[int]:
    midi = []
    for idx in range(len(chroma)):
        if chroma[idx] == 1:
            midi.append(idx)
    return midi


def set_binary_chroma(chroma: List[float], threshold: float) -> List[int]:
    return [1 if c > threshold else 0 for c in chroma]


def determine_roots(inpt: np.ndarray, model: ChordTransformer, mask: np.ndarray) -> List[int]:
    inpt = torch.tensor(inpt).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0).transpose(0, 1)
    logits = model(inpt, src_key_padding_mask=(mask == 0))
    return torch.argmax(logits, dim=-1).tolist()[0]


def preprocess_input(binary_chroma: List[List[int]]) -> np.ndarray:
    preprocessed = []
    duration = 0
    for i in range(len(binary_chroma) - 1):
        duration += 4
        if binary_chroma[i] != binary_chroma[i + 1]:
            preprocessed.append(binary_chroma[i])
            preprocessed[-1].append(duration / 16)
            duration = 0

    if binary_chroma[-1] == binary_chroma[-2]:
        preprocessed.append(binary_chroma[-1])
        preprocessed[-1].append((duration + 4) / 16)

    for _ in range(max_len - len(preprocessed)):
        preprocessed.append([-1 for _ in range(13)])

    return np.array(preprocessed, dtype=np.float32)