import os
import librosa
import numpy as np
from audio import extract_melody
from constants import NOTES, RATE
from torch.utils.data import Dataset
from typing import Tuple
note_to_idx = {note: i for i, note in enumerate(NOTES)}
idx_to_note = {i: note for i, note in enumerate(NOTES)}

def load_labels(file_path: str) -> dict[str, Tuple[str, np.ndarray, np.ndarray]]:
    with open(file_path, "r") as f:
        labels = f.readlines()

    return {label.split("|")[0]: (label.split("|")[1], None, None) for label in labels}


class MelodyHarmonyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data = load_labels(os.path.join(data_dir, "labels.txt"))

        for folder in os.listdir(data_dir):
            for file in os.listdir(os.path.join(data_dir, folder)):
                if file.startswith("mixture"):
                    name = folder.title()
                    label = self.data[name][0]

                    mixture, _ = librosa.load(os.path.join(data_dir, folder, file), sr=RATE)
                    harmony, _ = librosa.effects.hpss(mixture)
                    melody = extract_melody(mixture)
                    self.data[name] = (label, melody, harmony)


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, name: str):
        return self.data[name]