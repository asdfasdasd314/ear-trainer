import torch.nn as nn
from constants import CHUNK
import data
import librosa
import os
from torch.utils.data import DataLoader
import ml
import torch
from constants import RATE

from torch.nn.utils.rnn import pad_sequence

# Pad in the collate_fn (used in DataLoader)
def collate_fn(batch):
    chromas, labels = zip(*batch)
    chromas = pad_sequence(chromas, batch_first=True)  # Pads to longest in batch
    labels = pad_sequence(labels, batch_first=True)
    return chromas, labels
    

def main():
    chromas = data.load_chromas("musdb18hq/cache/chromas/train")
    labels = data.load_tonal_center_labels("musdb18hq/train/train_labels.txt")

    print("Done loading cached data")

    dataset = data.TonalCenterDataset(chromas, labels)

    print("Done loading dataset")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print("Done loading data")

    # Train model
    model = ml.TonalCenterModel(hidden_size=128, num_layers=2)
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ml.train_tonal_center_model(model, dataloader, loss_function, optimizer, num_epochs=10)

if __name__ == "__main__":
    main()