from typing import List, Tuple
import numpy as np
import librosa
import essentia.standard as es
import wave
import pyaudio
import threading
import os
from constants import RATE, CHUNK, FORMAT, CHANNELS

def load_song(song_path: str) -> np.ndarray:
    y, _ = librosa.load(os.path.join(song_path, "mixture.wav"), sr=RATE)
    return y


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


def spectrogram_to_bytes(S: np.ndarray) -> List[bytes]:
    waveform = librosa.griffinlim(S, hop_length=CHUNK // 2, n_fft=CHUNK, n_iter=32)
    waveform = waveform / np.max(np.abs(waveform))
    waveform = np.clip(waveform, -1.0, 1.0)
    return (waveform * 32767).astype(np.int16)


def save_audio(audio: List[bytes], filename: str):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio))


def extract_melody(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = es.EqloudLoader(filename=filename, sampleRate=RATE)
    audio = loader()
    pitch_extractor = es.PredominantPitchMelodia(
        frameSize=CHUNK,
        hopSize=CHUNK // 2,
        sampleRate=RATE,
        guessUnvoiced=True,
        filterIterations=3,
        minFrequency=55.0,
        maxFrequency=1760.0,
    )
    pitch_values, pitch_confidence = pitch_extractor(audio)
    pitch_times = np.linspace(0.0, len(audio) / RATE, len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times


def remove_f0(S: np.ndarray, pitches: np.ndarray, rate=RATE, fft_size=CHUNK, bandwidth_hz=30) -> np.ndarray:
    """
    Removes the fundamental frequency (F0) region from a transposed spectrogram.

    Args:
        S (np.ndarray): Transposed spectrogram (time_frames, freq_bins)
        pitches (np.ndarray): Detected pitches in Hz (length = time_frames)
        rate (int): Sample rate
        fft_size (int): FFT size used in STFT
        bandwidth_hz (int): Bandwidth to zero around pitch in Hz

    Returns:
        np.ndarray: Modified spectrogram with melody removed
    """
    mask = np.ones_like(S)
    bin_width = rate / fft_size
    bin_radius = int(bandwidth_hz / bin_width)
    num_harmonics = 1

    for i, pitch in enumerate(pitches):
        if pitch <= 0:
            continue

        for h in range(1, num_harmonics + 1):
            harmonic_freq = h * pitch
            pitch_bin = int(harmonic_freq / bin_width)
            start = max(0, pitch_bin - bin_radius)
            end = min(S.shape[1], pitch_bin + bin_radius)
            mask[i, start:end] = 0

    return S * mask


def synthesize_melody(pitches: np.ndarray) -> np.ndarray:
    hop_size = CHUNK // 2
    duration = len(pitches) * hop_size / RATE
    t = np.linspace(0, duration, len(pitches) * hop_size, endpoint=False)
    melody_wave = np.zeros_like(t)

    phase = 0.0  # Continuous phase accumulator

    for i, pitch in enumerate(pitches):
        if pitch > 0:
            start = i * hop_size
            end = start + hop_size
            if end > len(t):
                end = len(t)

            # Compute instantaneous frequency
            instantaneous_t = t[start:end]
            phase_increment = 2 * np.pi * pitch / RATE
            frame_phase = phase + np.cumsum(np.full(end - start, phase_increment))

            melody_wave[start:end] += 0.5 * np.sin(frame_phase)
            phase = frame_phase[-1]  # Update phase for continuity
        else:
            # If pitch == 0 (unvoiced), just hold silence or zero
            continue

    # Optional: Normalize to -1 to 1
    melody_wave = melody_wave / np.max(np.abs(melody_wave))
    return melody_wave
