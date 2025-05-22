import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from constants import CHUNK

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = (CHUNK + 1) + 1 # Number of frequency bins + 1 for the melody
        self.output_size = 12 # number of notes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.softmax(out)
        return out
    

def train_model(model: LSTM, loader: DataLoader, loss_function: nn.NLLLoss, optimizer: torch.optim.Optimizer, num_epochs: int):
    while num_epochs > 0:
        model.zero_grad()

        for (label, melody, harmony) in loader:
            if melody.ndim == 1:
                melody = melody.unsqueeze(1)

            inputs = torch.cat((harmony, melody), dim=1)
            output = model(inputs)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

        num_epochs -= 1


def save_model(model: LSTM):
    model_scripted = torch.jit.script(model)
    model_scripted.save("models/model.pt")


def load_model() -> LSTM:
    model: LSTM = torch.jit.load("models/model.pt")
    model.eval()
    return model


def test_model(model: LSTM, loader: DataLoader):
    accuracy = 0
    for inputs, labels in loader:
        output = model(inputs)
        if output.argmax(dim=1) == labels:
            accuracy += 1

    return accuracy / len(loader.dataset)
