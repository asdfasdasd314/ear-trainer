import os
import librosa
import numpy as np
import torch
from audio import extract_melody
from constants import CHUNK, KEYS, NOTES, RATE
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

key_to_idx = {key: i for i, key in enumerate(KEYS)}
idx_to_key = {i: key for i, key in enumerate(KEYS)}

note_to_idx = {note: i for i, note in enumerate(NOTES)}
idx_to_note = {i: note for i, note in enumerate(NOTES)}


def save_np_arrays(path: str, data: dict[str, np.ndarray]):
    for key, array in data.items():
        np.save(os.path.join(path, f"{key}.npy"), array)


def frequency_to_12_tone(frequency: float) -> int:
    if frequency == 0:
        return -1

    above_a4 = 12 * np.log2(frequency / 440.0)

    # Start on C, so shift up 3
    above_c4 = above_a4 + 3

    # Wrap around
    return above_c4 % 12


def load_tonal_center_labels(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        return {line.split("|")[0].strip(): note_to_idx[line.split("|")[1].split(" ")[0]] for line in f.readlines()}


def load_key_labels(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        return {line.split("|")[0].strip(): key_to_idx[line.split("|")[1].split(" ")[0]] for line in f.readlines()}


def load_chromas(path: str) -> dict[str, np.ndarray]:
    files = os.listdir(path)
    chromas = {}
    for file in files:
        if file.endswith(".npy"):
            chromas[file[:-4]] = np.load(os.path.join(path, file))

    return chromas


def partition_chromas(chromas: np.ndarray) -> List[np.ndarray]:
    if chromas.shape[1] < 2000:
        return [chromas.T]
    
    partitions = []
    for i in range(0, chromas.shape[1], 2000):
        partitions.append(chromas[:, i:i+2000].T)

    return partitions


class TonalCenterDataset(Dataset):

    def __init__(self, chromas: dict[str, np.ndarray], labels: dict[str, int]):
        assert len(chromas) == len(labels)

        self.chromas = []
        self.labels = []

        for key in chromas.keys():
            chroma = chromas[key]
            partitions = partition_chromas(chroma)
            self.chromas.extend(partitions)

            for partition in partitions:
                partition_labels = []
                for _ in partition:
                    partition_labels.append(labels[key])

                self.labels.append(partition_labels)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chroma = torch.tensor(self.chromas[index])
        label = torch.tensor(self.labels[index])

        return chroma, label


    def __len__(self):
        return len(self.chromas)
