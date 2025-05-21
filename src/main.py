from typing import Tuple
import numpy as np
import librosa
import pyaudio
import threading

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

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


def collect_microphone_audio() -> np.ndarray:
    pass


def parse_harmony(harmonic: np.ndarray, f0: np.ndarray) -> np.ndarray:
    D = librosa.stft(harmonic)
    frequencies = np.abs(D)


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

# Convert frames to normalized float array in numpy
audio_data = np.frombuffer(frames[0], dtype=np.int16)
audio_data = audio_data / np.max(np.abs(audio_data))

res: Tuple[np.ndarray, np.ndarray] = librosa.effects.hpss(audio_data, RATE)

percussive = res[0]
harmonic = res[1]


def main():
    audio_data = collect_microphone_audio()
    res: Tuple[np.ndarray, np.ndarray] = librosa.effects.hpss(audio_data, RATE)

    harmonic = res[1]

    f0 = librosa.pyin(harmonic, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    harmony = parse_harmony(harmonic, f0)

if __name__ == "__main__":
    main()