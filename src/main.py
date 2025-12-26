from collections import Counter
import threading

import matplotlib.pyplot as plt
import pyaudio
from constants import ATTACK_HISTORY_LENGTH, REGULAR_HISTORY_LENGTH, ATTACK_THRESHOLD_1, ATTACK_THRESHOLD_2, HARMONIC_REDUCTION, FORMAT, CHANNELS, HOP_SIZE, IDX_TO_NOTE, RATE, CHUNK, MAX_VALUE, BUCKET_SIZE, MIN_FREQ, MAX_FREQ, STRENGTH_THRESHOLD_1, STRENGTH_THRESHOLD_2, PROXIMITY_THRESHOLD
import numpy as np
from audio import remove_harmonics, peaks_to_midi, cluster_peaks, locate_peaks, compute_weighted_chroma, chroma_to_midi, determine_roots, preprocess_input, construct_chord_symbol, mask_noise
import time
from queue import Queue
import librosa

from entropy import determine_chord_symbol

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

audio_queue = Queue()

def get_frames():
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put_nowait(data)

stream.start_stream()
input_thread = threading.Thread(target=get_frames)
input_thread.start()

# # Initialize plot once before the loop
# plt.ion()  # Turn on interactive mode
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# ax1.set_xlim(0, MAX_FREQ - MIN_FREQ)
# ax1.set_ylim(0, 200)  # Fix y-axis scale from 0 to 100
# ax1.set_title('Spectrum')
# ax1.set_xlabel('Frequency Bin')
# ax1.set_ylabel('Magnitude')
# ax2.set_title('Peaks')
# ax2.set_xlabel('Frequency Bin')
# ax2.set_ylabel('Magnitude')
# plt.tight_layout()

note_history  = []
bass_history = []
prev_chord = ""

local_copy = bytearray()

# Shape: (MAX_FREQ - MIN_FREQ,) - frequency values corresponding to each spectrum bin
freqs = np.fft.rfftfreq(BUCKET_SIZE, 1.0 / RATE)[MIN_FREQ:MAX_FREQ]

# Convert frequencies to MIDI note numbers: numpy array
midi = np.round(69 + 12 * np.log2(freqs / 440.0))

chroma = midi % 12

while True:
    if audio_queue.qsize() > 0:
        chunk = audio_queue.get()
        local_copy.extend(chunk)
    if len(local_copy) > BUCKET_SIZE:
        samples = np.frombuffer(local_copy[:BUCKET_SIZE], dtype=np.int16)
        local_copy = local_copy[HOP_SIZE:] # Allows for overlap between frames, although does copy the memory every time (kind of slow)

        bucket = samples.astype(np.float32) / MAX_VALUE

        spectrum = np.abs(np.fft.rfft(bucket, n=BUCKET_SIZE))[MIN_FREQ:MAX_FREQ]

        peaks = locate_peaks(spectrum, freqs, ATTACK_THRESHOLD_1)
        attack_detected = len(peaks) > 0
        if not attack_detected:
            peaks = locate_peaks(spectrum, freqs, STRENGTH_THRESHOLD_1)
        peaks = cluster_peaks(peaks, PROXIMITY_THRESHOLD)
        peaks = [[peak[0] * (1 + (HARMONIC_REDUCTION / peak[1]) ** 2), peak[1], peak[2], peak[3]] for peak in peaks] # Account for lower frequencies being not as loud
        peaks = np.array(remove_harmonics(peaks))
        if not attack_detected:
            peaks = [peak for peak in peaks if peak[0] > STRENGTH_THRESHOLD_2]
        else:
            peaks = [peak for peak in peaks if peak[0] > ATTACK_THRESHOLD_2]
        peaks = peaks_to_midi(peaks, freqs)
        notes = [int(chroma[int(midi)]) for midi in peaks]

        bass = notes[0] if len(notes) > 0 else None
        notes = set(notes)

        note_history.append(notes)
        bass_history.append(bass)

        if attack_detected:
            history_length = ATTACK_HISTORY_LENGTH
        else:
            history_length = REGULAR_HISTORY_LENGTH

        if len(notes) >= history_length:
            # Pick the notes that all have in common
            input_notes = note_history[-history_length:]
            common_notes = set.intersection(*input_notes)
            most_common_bass = Counter(bass_history[-history_length:]).most_common(1)[0][0]
            if len(common_notes) > 0:
                binary_chroma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for note in common_notes:
                    binary_chroma[note] = 1
                chord_symbol = determine_chord_symbol(binary_chroma, bass=most_common_bass)
                if chord_symbol != "N/A" and prev_chord != chord_symbol:
                    print(f"Chord: {chord_symbol}")
                    prev_chord = chord_symbol