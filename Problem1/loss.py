import numpy as np

import torch
from torch import nn
from torch.autograd import grad


class JensenShannonDivergence(nn.Module):
    def __init__(self, bias=2.0, cost=0.5, eps=1e-8):
        super(JensenShannonDivergence, self).__init__()
        self.bias = torch.log(torch.tensor(bias))
        self.cost = cost
        self.eps = eps

    def forward(self, p, q):
        # Expected values
        expected_p = torch.log(p + self.eps).mean()
        expected_q = torch.log(1 - q + self.eps).mean()

        # Negative loss since torch minimizes
        loss = - (self.bias + self.cost * expected_p + self.cost * expected_q)

        return loss


class WasserteinDistance(nn.Module):
    def __init__(self, lda=10, eps=1e-8):
        super(WasserteinDistance, self).__init__()
        self.eps = eps
        self.lda = lda

    def forward(self, p, q, z_in, z_out):
        # Expected values
        expected_p = torch.mean(p + self.eps)
        expected_q = torch.mean(q + self.eps)

        # Gradient penalty to ensure function is 1-lipschitz
        gradients = grad(outputs=z_out, inputs=z_in, grad_outputs=torch.ones_like(z_out),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()

        # Negative loss since torch minimizes
        loss = - (expected_p - expected_q - self.lda * gradient_penalty)

        return loss
