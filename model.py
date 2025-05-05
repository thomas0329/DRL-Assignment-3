import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=3):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_eps', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(epsilon_out.ger(epsilon_in))
        self.bias_eps.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, sigma_init=0.5):
#         super(NoisyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
#         self.register_buffer('weight_eps', torch.empty(out_features, in_features))

#         self.bias_mu = nn.Parameter(torch.empty(out_features))
#         self.bias_sigma = nn.Parameter(torch.empty(out_features))
#         self.register_buffer('bias_eps', torch.empty(out_features))

#         self.sigma_init = sigma_init
#         self.reset_parameters()
#         self.reset_noise()

#     def reset_parameters(self):
#         mu_range = 1 / math.sqrt(self.in_features)
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.sigma_init * mu_range)

#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.sigma_init * mu_range)
        

#     def reset_noise(self):
#         epsilon_in = self._scale_noise(self.in_features)
#         epsilon_out = self._scale_noise(self.out_features)

#         self.weight_eps.copy_(epsilon_out.ger(epsilon_in))
#         self.bias_eps.copy_(epsilon_out)

#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         return x.sign() * x.abs().sqrt()

#     def forward(self, x):
#         if self.training:
#             weight = self.weight_mu + self.weight_sigma * self.weight_eps
#             bias = self.bias_mu + self.bias_sigma * self.bias_eps
#         else:
#             weight = self.weight_mu
#             bias = self.bias_mu
#         return F.linear(x, weight, bias)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.noisy1 = NoisyLinear(6272, 1024)
        self.noisy2 = NoisyLinear(1024, output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            self.noisy1,
            # nn.Linear(3136, 512),
            nn.ReLU(),
            self.noisy2
            # nn.Linear(512, output_dim)
        )
        

    def forward(self, x):
        return self.net(x)
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, use_noisy=False):
        super(DuelingDQN, self).__init__()
        self.use_noisy = use_noisy

        # Feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Dummy pass to get flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_dim)
            feature_dim = self.feature(dummy).shape[1]

        Linear = NoisyLinear if use_noisy else nn.Linear

        # Value stream
        self.value_stream = nn.Sequential(
            Linear(feature_dim, 512),
            nn.ReLU(),
            Linear(512, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            Linear(feature_dim, 512),
            nn.ReLU(),
            Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        if self.use_noisy:
            for module in list(self.value_stream) + list(self.advantage_stream):
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
