import threading

import pyaudio
from constants import FORMAT, CHANNELS, RATE, CHUNK, MAX_VALUE, BUCKET_SIZE, MIN_FREQ, MAX_FREQ, STRENGTH_THRESHOLD
import numpy as np
from audio import get_microphone_audio, compute_weighted_chroma, set_binary_chroma, chroma_to_midi, determine_roots, preprocess_input, construct_chord_symbol
from transformer import ChordTransformer, d_model, n_heads, num_layers, input_dim, num_classes, models_base
import torch

def sample_chroma(seconds: float=float("inf"), stream: pyaudio.Stream=None):
    frames = get_microphone_audio(seconds=seconds, stream=stream)

    raw_audio = b"".join(frames)
    samples = np.frombuffer(raw_audio, dtype=np.int16)

    normalized = samples.astype(np.float32) / MAX_VALUE

    num_buckets = len(normalized) // BUCKET_SIZE + 1
    buckets = [normalized[i * BUCKET_SIZE:(i + 1) * BUCKET_SIZE] for i in range(num_buckets)]

    spectra = [np.abs(np.fft.rfft(bucket, n=BUCKET_SIZE))[MIN_FREQ:MAX_FREQ] for bucket in buckets]
    freqs = np.fft.rfftfreq(BUCKET_SIZE, 1.0 / RATE)[MIN_FREQ:MAX_FREQ]

    midi = np.round(69 + 12 * np.log2(freqs / 440.0))
    chroma = midi % 12

    return [compute_weighted_chroma(spectrum, chroma) for spectrum in spectra] 

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

model = ChordTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers, input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load(str(models_base / "model31.pth")))
model.eval()

print("Begin playing...")

weighted_chroma = sample_chroma(stream=stream)
binary_chroma = [set_binary_chroma(chroma, STRENGTH_THRESHOLD) for chroma in weighted_chroma]
print(weighted_chroma)
print(binary_chroma)

stream.close()

inpt = preprocess_input(binary_chroma)
mask = np.array([1 if i != -1 else 0 for i in inpt[:, 0]])
roots = determine_roots(inpt, model, mask)
chord_symbols = []
for i in range(len(inpt)):
    if mask[i] == 1:
        chord_symbols.append(construct_chord_symbol(roots[i], chroma_to_midi(binary_chroma[i])))
    else:
        break

print(chord_symbols)