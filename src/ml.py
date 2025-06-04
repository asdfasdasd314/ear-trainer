import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from constants import CHUNK, NOTES

class TonalCenterModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(TonalCenterModel, self).__init__()
        self.input_size = len(NOTES) # number of chroma
        self.output_size = len(NOTES) # number of notes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, self.output_size)


    def forward(self, x):
        # Potentially needs to be fixed, I don't really know
        output, _ = self.lstm(x)
        
        # Aggregate over time to predict for the entire sequence
        pooled = output.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
    

def train_tonal_center_model(model: TonalCenterModel, loader: DataLoader, loss_function: nn.NLLLoss, optimizer: torch.optim.Optimizer, num_epochs: int):
    while num_epochs > 0:
        for chroma, label in loader:
            optimizer.zero_grad()
            output = model(chroma)
            output = output.permute(0, 2, 1)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {num_epochs} loss: {loss.item()}")
        num_epochs -= 1


def save_model(model: nn.Module):
    model_scripted = torch.jit.script(model)
    model_scripted.save("models/model.pt")


def load_tonal_center_model() -> TonalCenterModel:
    model: TonalCenterModel = torch.jit.load("models/model.pt")
    model.eval()
    return model


def test_tonal_center_model(model: TonalCenterModel, loader: DataLoader):
    accuracy = 0
    for chroma, label in loader:
        output = model(chroma)
        if output.argmax(dim=1) == label:
            accuracy += 1

    return accuracy / len(loader.dataset)
