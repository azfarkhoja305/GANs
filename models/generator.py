
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,z_sz=128,p_sz=1024):
        super().__init__()
        self.p_sz=p_sz
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(z_sz,p_sz, bias=False)
        self.bn1d = nn.BatchNorm1d(p_sz)
        self.body = nn.Sequential(
            # 246 x 2 x 2
            nn.PixelShuffle(upscale_factor=2),
            # 64 x 4 x 4
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # 256 x 4 x 4
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            # 64 x 8 x 8
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # 256 x 8 x 8
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            # 64 x 16 x 16
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # 256 x 16 x 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(upscale_factor=2),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # 64 x 32 x 32
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False),
            # 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self,x):
        x = x.squeeze()
        x = self.bn1d(self.relu(self.fc(x)))
        x = x.view(-1,self.p_sz//4,2,2)
        x = self.body(x)
        return x
