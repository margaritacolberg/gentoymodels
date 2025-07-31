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

    def __init__(self, input_size, hidden_size, output_size, num_hidden):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
