import torch.nn as nn


class MLP_ddpm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, freqs=3):
        super(MLP_ddpm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, features):
        out = self.fc1(features)
        out = self.act(out)
        out = self.fc2(out)
        return out


class MLP_adjoint(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_adjoint, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.GELU()

    def forward(self, features):
        out = self.fc1(features)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out
