import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.elu = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc_mean = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.fc_log_variance = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.fc_decoder = nn.Linear(in_features=self.latent_dim, out_features=256)
        self.encoding = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU()
        )
        self.decoding_part1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=4),
            nn.ELU()
        )
        self.decoding_part2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2),
            nn.ELU()
        )
        self.decoding_part3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=2)
        )

    def encode(self, x):
        x = self.encoding(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        log_variance = self.fc_log_variance(x)
        return mean, log_variance

    def reparameterize(self, mu, log_variance):
        sigma = torch.exp(0.5*log_variance)
        e = torch.randn_like(log_variance)
        z = e.mul(sigma)
        z.add_(mu)
        return z

    def decode(self, z):
        x = self.elu(self.fc_decoder(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.decoding_part1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.decoding_part2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.decoding_part3(x)
        return x

    def forward(self, x):
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance)
        x_tilde = self.decode(z)
        return x_tilde, mean, log_variance  # need to return mu and log_variance in order to compute loss

