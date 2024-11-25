import torch
import torch.nn as nn

class DRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)
        self.d = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        term1 = x + (self.b * torch.square(torch.sin(self.a * x)) / self.a)
        term2 = self.c * torch.cos(self.a * x)
        term3 = self.d * torch.tanh(self.b * x)
        return term1 + term2 + term3
