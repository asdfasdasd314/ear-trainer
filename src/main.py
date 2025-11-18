import threading

import pyaudio
from constants import FORMAT, CHANNELS, RATE, CHUNK, MAX_VALUE, BUCKET_SIZE, MIN_FREQ, MAX_FREQ, STRENGTH_THRESHOLD
import numpy as np
from audio import get_microphone_audio, compute_weighted_chroma, set_binary_chroma, chroma_to_midi, determine_roots, preprocess_input, construct_chord_symbol, mask_noise
from transformer import ChordTransformer, d_model, n_heads, num_layers, input_dim, num_classes, models_base
import torch
import time

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

model = ChordTransformer(d_model=d_model, n_heads=n_heads, num_layers=num_layers, input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load(str(models_base / "model31.pth")))
model.eval()

model_context = [] # Array of pre-processed inputs to be passed directly to model

audio_frames = []

def get_frames():
    stream.start_stream()
    while True:
        data = stream.read(CHUNK)
        audio_frames.append(data)

input_thread = threading.Thread(target=get_frames)
input_thread.start()

last_eval = time.time()

while True:
    if len(audio_frames) > RATE:
        raw_audio = b"".join(audio_frames)
        audio_frames = [] # Reset buffer

        samples = np.frombuffer(raw_audio, dtype=np.int16)

        normalized = samples.astype(np.float32) / MAX_VALUE

        # I believe the reason we split into buckets is because FFT takes a fixed length input
        num_buckets = len(normalized) // BUCKET_SIZE + 1
        buckets = [normalized[i * BUCKET_SIZE:(i + 1) * BUCKET_SIZE] for i in range(num_buckets)]

        spectra = [np.abs(np.fft.rfft(bucket, n=BUCKET_SIZE))[MIN_FREQ:MAX_FREQ] for bucket in buckets]

        # Expand spectra to 2D array
        spectra = np.array(spectra)

        # Compute average of spectra and pass as STRENGTH_THRESHOLD to mask_noise
        strength_threshold = np.mean(spectra)

        # Mask out noise
        spectra = mask_noise(spectra, strength_threshold)

        # Compute the actual frequency values (in Hz) for each bin in the spectrum
        # Shape: (12524,) - frequency values corresponding to each spectrum bin
        freqs = np.fft.rfftfreq(BUCKET_SIZE, 1.0 / RATE)[MIN_FREQ:MAX_FREQ]

        # Convert frequencies to MIDI note numbers: numpy array
        midi = np.round(69 + 12 * np.log2(freqs / 440.0))
        # Convert MIDI to chroma (0-11): numpy array
        chroma = midi % 12

        # Compute weighted chroma for each spectrum: list of numpy arrays, each of shape (12,)
        weighted_chroma = [compute_weighted_chroma(spectrum, chroma) for spectrum in spectra]
        
        # Convert to binary chroma: list of lists (each inner list has 12 ints)
        # The STRENGTH_THRESHOLD is determined by measuring ambient sound intensity versus when sound is actually played
        binary_chroma = [set_binary_chroma(chroma, STRENGTH_THRESHOLD) for chroma in weighted_chroma]

        model_context.append(preprocess_input(binary_chroma))
        
    if time.time() - last_eval > 2.0:
        # Preprocess for model input: numpy array
        inpt = model_context
        mask = np.array([1 if i != -1 else 0 for i in inpt[:, 0]])
        roots = determine_roots(inpt, model, mask)
        chord_symbols = []

        for i in range(len(inpt)):
            if mask[i] == 1:
                chord_symbols.append(construct_chord_symbol(roots[i], chroma_to_midi(binary_chroma[i])))
            else:
                break

        print(chord_symbols)

        last_eval = time.time()