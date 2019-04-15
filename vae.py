import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.elu = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc_mean = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.fc_log_variance = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.fc_decoder = nn.Linear(in_features=self.latent_dim, out_features=256)

    def encode(self, x):
        x = self.pooling(self.elu(nn.Conv2d(1, 32, kernel_size=(3, 3))(x)))
        x = self.pooling(self.elu(nn.Conv2d(32, 64, kernel_size=(3, 3))(x)))
        x = self.elu(nn.Conv2d(64, 256, kernel_size=(5, 5))(x))
        x = x.view(-1, 256)
        mean = self.fc_mean(x)
        log_variance = self.fc_log_variance(x)
        return mean, log_variance

    def reparameterize(self, mu, log_variance):
        sigma = torch.exp(0.5*log_variance)
        e = torch.zeros(sigma.size()).normal_()
        e = Variable(e)
        z = e.mul(sigma)
        z.add_(mu)
        return z

    def decode(self, z):
        x = self.elu(self.fc_decoder(z))
        x = x.view(-1, 256, 1, 1)
        x = self.elu(nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4))(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.elu(nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2))(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.elu(nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2))(x))
        x = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))(x)
        return x

    def forward(self, x):
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance)
        x_tilde = self.decode(z)
        return x_tilde, mean, log_variance  # need to return mu and log_variance in order to compute loss

