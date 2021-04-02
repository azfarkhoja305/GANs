import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            # 32 x 16 x 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            # 64 x 8 x 8
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            # 128 x 4 x 4
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),
            # 256 x 1 x 1
        )
        self.linear = nn.Linear(256, 1, bias=False)

    def forward(self, x):
        x = self.body(x)
        x = self.linear(x.view(x.size(0), -1))
        return x
