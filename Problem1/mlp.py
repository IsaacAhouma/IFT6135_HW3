from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers=[], activation='relu', output=None):
        super(MLP, self).__init__(),
        # Architecture parameters
        self.architecture = [in_features] + layers + [out_features]
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
        })
        self.output = self.activations[output] if output else output

        # Architecture
        layers = []
        for i in range(0, len(self.architecture) - 1):
            layers.append(nn.Linear(self.architecture[i], self.architecture[i + 1]))
            layers.append(self.activations[activation])
        self.layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self.layers(x)
        if self.output is not None:
            x = self.output(x)

        return x


if __name__ == '__main__':
    mlp = MLP(2, 1, [5, 5, 5], 'relu', 'sigmoid')
    print(mlp)
