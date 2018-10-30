import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        layer.bias.data.fill_(0.)


def build_mlp(in_dim: tuple, out_dim: tuple, layers: list):
    net = [nn.Linear(in_dim[0], layers[0])]

    for i in range(len(layers) - 1):
        net.append(nn.ReLU())
        net.append(nn.Linear(layers[i], layers[i + 1]))

    net.append(nn.ReLU())
    net.append(nn.Linear(layers[-1], out_dim[0]))

    net = nn.Sequential(*net).to(device)
    net.apply(init_weights)

    return net
