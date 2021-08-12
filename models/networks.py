import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, inputs, hiddens, out, activation, lin_type='regular'): 
        super(MLP, self).__init__()
        activation = self._select_activation(activation)

        lin_lay = self._select_lin(lin_type)
        layers = [lin_lay(inputs, hiddens[0]), activation()]
        for (in_d, out_d) in zip(hiddens[:-1], hiddens[1:]):
            
            layers = layers + [lin_lay(in_d, out_d)]
            layers = layers + [activation()]
        layers = layers + [lin_lay(hiddens[-1], out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _select_activation(self, act):
        if act == 'tanh':
            return nn.Tanh
        elif act == 'relu':
            return nn.ReLU
        elif act == 'sigmoid':
            return nn.Sigmoid

    def _select_lin(self, lin):
        if lin == 'regular':
            return nn.Linear
        elif lin == 'reg-no-bias':
            return lambda input, output: nn.Linear(input, output, bias=False)



if __name__ == '__main__':
    pass