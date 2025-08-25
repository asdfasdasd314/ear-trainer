import pyaudio
import threading
from constants import FORMAT, CHANNELS, RATE, CHUNK, MAX_VALUE, BUCKET_SIZE, MIN_FREQ, MAX_FREQ
import numpy as np
from audio import compute_weighted_chroma, set_binary_chroma, chroma_to_midi, determine_roots, preprocess_input, construct_chord_symbol
from transformer import ChordTransformer, d_model, n_heads, num_layers, input_dim, num_classes, models_base
import torch

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

read = [] # For some reason the signal has to be updating a list

def process_input():
    while True:
        inp = input()
        if inp == "q":
            read.append(True)
            break

read_input = threading.Thread(target=process_input)
read_input.start()

print("Listening...")

frames = []
stream.start_stream()
while True:
    data = stream.read(CHUNK)
    frames.append(data)
    if len(read) > 0:
        break

stream.stop_stream()
stream.close()

print("Processing...")

raw_audio = b"".join(frames)
samples = np.frombuffer(raw_audio, dtype=np.int16)

normalized = samples.astype(np.float32) / MAX_VALUE

num_buckets = len(normalized) // BUCKET_SIZE + 1
buckets = [normalized[i * BUCKET_SIZE:(i + 1) * BUCKET_SIZE] for i in range(num_buckets)]

spectra = [np.abs(np.fft.rfft(bucket, n=BUCKET_SIZE))[MIN_FREQ:MAX_FREQ] for bucket in buckets]
freqs = np.fft.rfftfreq(BUCKET_SIZE, 1.0 / RATE)[MIN_FREQ:MAX_FREQ]

midi = np.round(69 + 12 * np.log2(freqs / 440.0))
chroma = midi % 12

model = ChordTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers, input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load(str(models_base / "model31.pth")))
model.eval()

weighted_chroma = [compute_weighted_chroma(spectrum, chroma) for spectrum in spectra] 
binary_chroma = [set_binary_chroma(chroma, strength_threshold) for chroma in weighted_chroma]
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