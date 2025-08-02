import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden=1,
                 activation='relu'):
        super().__init__()

        # choose activation
        act_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        }
        act = act_map.get(activation.lower(), nn.ReLU())

        layers = [nn.Linear(input_size, hidden_size), act]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), act]
        layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
