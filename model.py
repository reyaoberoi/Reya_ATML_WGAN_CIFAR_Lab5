import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, z_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),

            nn.Linear(256,512),
            nn.ReLU(),

            nn.Linear(512,1024),
            nn.ReLU(),

            nn.Linear(1024,3*32*32),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.net(x)
        return x.view(-1,3,32,32)