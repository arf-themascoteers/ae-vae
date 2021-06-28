import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, 2)
        self.log_var = nn.Linear(64, 2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        mean = self.mean(x)
        var = self.log_var(x)
        return mean, var, self.reparameterization(mean, var)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z

