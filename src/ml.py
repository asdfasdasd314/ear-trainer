import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
    

def train_model(model: LSTM, data: torch.Tensor, loss_function: nn.NLLLoss, optimizer: torch.optim.Optimizer, num_epochs: int):
    while num_epochs > 0:
        model.zero_grad()

        for i in range(len(data) - 1):
            x = data[i]
            y = data[i+1]
            output = model(x)
            loss = loss_function(output, y)
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


def test_model(model: LSTM, data: torch.Tensor):
    accuracy = 0
    for i in range(len(data) - 1):
        x = data[i]
        y = data[i+1]
        output = model(x)
        if output.argmax(dim=1) == y:
            accuracy += 1

    return accuracy / len(data)
