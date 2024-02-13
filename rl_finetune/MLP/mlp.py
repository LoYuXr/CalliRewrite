import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_size, output_size, n_hidden=1, hidden_dim=256, first_dim=0, add_tanh=False):
        super().__init__()

        first_dim = max(hidden_dim, first_dim)
        layers = [nn.Linear(input_size, first_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(first_dim, hidden_dim))
            first_dim = hidden_dim
            layers.append(nn.ReLU())
        layers.append(nn.Linear(first_dim, output_size))
        if add_tanh:
            layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: tensor of shape [batch_size, input_size]
        :return: logits: tensor of shape [batch_size, n_classes]
        """
        x = x.view(len(x), -1)  # flatten
        return self.mlp.forward(x)


class FourierMLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_hidden=1,
                 hidden_dim=256,
                 sigma=1.0,
                 fourier_dim=256,
                 train_B=False,
                 concatenate_fourier=False,
                 add_tanh=False):
        super().__init__()

        # create B
        b_shape = (input_size, fourier_dim // 2)
        self.sigma = sigma
        self.B = nn.Parameter(torch.normal(torch.zeros(*b_shape), torch.full(b_shape, sigma)))
        self.B.requires_grad = train_B

        self.concatenate_fourier = concatenate_fourier
        if self.concatenate_fourier:
            mlp_input_dim = fourier_dim + input_size
        else:
            mlp_input_dim = fourier_dim

        # create rest of the network
        layers = [nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size))
        if add_tanh:
            layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: tensor of shape [batch_size, input_size]
        :return: logits: tensor of shape [batch_size, n_classes]
        """
        x = x.view(len(x), -1)  # flatten
        # create fourier features
        proj = (2 * np.pi) * torch.matmul(x, self.B)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.concatenate_fourier:
            ff = torch.cat([x, ff], dim=-1)
        return self.mlp.forward(ff)


class SineLayer(nn.Module):
    # See SIREN sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, input_size, output_size, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = input_size
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=1, hidden_dim=256, outermost_linear=True,
                 first_omega_0=30.0, hidden_omega_0=30.0, add_tanh=False):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(input_size, hidden_dim, is_first=True, omega_0=first_omega_0))

        for i in range(n_hidden - 1):
            self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, output_size)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / hidden_omega_0,
                                             np.sqrt(6 / hidden_dim) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_dim, output_size, is_first=False, omega_0=hidden_omega_0))
        if add_tanh:
            self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
