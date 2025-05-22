from typing import List, Tuple
import numpy as np
import librosa
import essentia.standard as es
import wave
import pyaudio
import threading
from scipy.ndimage import gaussian_filter
from constants import RATE, CHUNK, FORMAT, CHANNELS

def collect_microphone_audio() -> List[bytes]:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    read = []

    def process_input():
        while True:
            inp = input()
            if inp == "q":
                read.append(True)
                break

    read_input = threading.Thread(target=process_input)
    read_input.start()

    frames = []
    stream.start_stream()
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if len(read) > 0:
            break

    stream.stop_stream()
    stream.close()

    return frames


def save_audio(audio: List[bytes], filename: str):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio))


def extract_melody(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = es.EqloudLoader(filename=filename, sampleRate=RATE)
    audio = loader()
    pitch_extractor = es.PredominantPitchMelodia(frameSize=CHUNK, hopSize=CHUNK // 2)
    pitch_values, pitch_confidence = pitch_extractor(audio)
    pitch_times = np.linspace(0.0, len(audio) / RATE, len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times


def create_melody_mask(S: np.ndarray, pitches: np.ndarray, sr=RATE, n_fft=CHUNK, width=3):
    """
    Create a mask that targets the melody frequencies in S
    S: (F, T) spectrogram
    pitches: (T,) pitch contour in Hz
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = np.zeros_like(S)

    for t, pitch in enumerate(pitches):
        if pitch > 0:
            idx = np.argmin(np.abs(freqs - pitch))
            start = max(0, idx - width)
            end = min(len(freqs), idx + width + 1)
            mask[start:end, t] = 0.5  # Keep some harmonic texture
    return mask


def extract_harmony(S: np.ndarray, melody: np.ndarray) -> np.ndarray:
    melody_mask = create_melody_mask(S, melody)

    # Smooth mask
    melody_mask = np.clip(melody_mask, 0.0, 1.0)

    # Apply complementary mask for harmony
    harmony_mask = 1.0 - melody_mask

    # Smoothed spectrograms
    S_harmony = S * harmony_mask

    return S_harmony


def synthesize_melody(pitches: np.ndarray):
    duration = len(pitches) * (CHUNK // 2) / RATE
    t = np.linspace(0, duration, len(pitches) * (CHUNK // 2), endpoint=False)
    melody_wave = np.zeros_like(t)

    for i, pitch in enumerate(pitches):
        if pitch > 0:
            start = i * (CHUNK // 2)
            end = start + (CHUNK // 2)
            if end > len(t):
                end = len(t)
            phase = 2 * np.pi * pitch * t[start:end]
            melody_wave[start:end] += 0.5 * np.sin(phase)

    # Optional: Normalize
    melody_wave = melody_wave / np.max(np.abs(melody_wave))
    return melody_wave