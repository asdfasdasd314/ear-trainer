from constants import CHUNK, IDX_TO_NOTE, RATE, BLUES_PROGRESSIONS
import random
from generate_blues import BluesProgression
import plot
import ml
import torch
import audio
from scipy.io import wavfile
import librosa
import numpy as np
from torch.utils.data import DataLoader

def main():
    for i in range(250):
        tonal_center_idx = random.randint(0, 11)
        progression_idx = random.randint(0, len(BLUES_PROGRESSIONS) - 1)
        loops = random.randint(6, 16)
        tempo = random.randint(60, 160)
        blues_progression = BluesProgression(tonal_center_idx, progression_idx, loops, tempo)
        blues_progression.output_to_file(f"data/chords/Blues/{i}.lab")

if __name__ == "__main__":
    main()