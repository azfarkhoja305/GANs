import os
import sys
from pathlib import Path
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary
from datasets import ImageDataset
from utils.utils import check_gpu, display_images
from models.generator import Generator

# from models.discriminator import Discriminator
from models.ViT_discriminator import Discriminator
from types import SimpleNamespace

device = check_gpu()
print(f"Using device: {device}")

dataset = ImageDataset("cifar_10", batch_sz=256)

# display_images(dataset.train_loader, max_idx=256)

Gen = Generator().to(device)
summary(Gen, (128, 1))

# Dis = Discriminator().to(device)
args = SimpleNamespace(**{"d_depth": 7, "df_dim": 384, "img_size": 32, "patch_size": 8})
Dis = Discriminator(args).to(device)
summary(Dis, (3, 32, 32))

loss_fn = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(64, 128, device=device)
real_label = 1.0
fake_label = 0.0
lr, beta1 = 3e-4, 0
optG = optim.AdamW(Gen.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.AdamW(Dis.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

# Number of training epochs
num_epochs = 50

# Commented out IPython magic to ensure Python compatibility.
for epoch in range(num_epochs):
    for i, data in enumerate(dataset.train_loader):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        Dis.zero_grad()
        # Format batch
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = Dis(real).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss_fn(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = torch.sigmoid(output).mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 128, 1, 1, device=device)
        # Generate fake image batch with G
        fake = Gen(noise)
        label = torch.full_like(label, fake_label)
        # Classify all fake batch with D
        output = Dis(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss_fn(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = torch.sigmoid(output).mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Gen.zero_grad()
        label = torch.full_like(
            label, real_label
        )  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = Dis(fake).view(-1)
        # Calculate G's loss based on this output
        errG = loss_fn(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = torch.sigmoid(output).mean().item()
        # Update G
        optG.step()

        #         # Output training stats
        #         if (i+1) %100 == 0:
        #             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        # #                 % (epoch, num_epochs, i, len(dataset.train_loader),
        #                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (
            (epoch == num_epochs - 1) and (i == len(dataset.train_loader) - 1)
        ):
            with torch.no_grad():
                fake = Gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

_, axs = plt.subplots(1, 2, figsize=(15, 15))
display_images(dataset.train_loader, ax=axs[0], max_idx=256)
display_images(img_list, ax=axs[1], max_idx=256)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""Calculating FID Score"""

from utils.torch_fid_score import get_fid

stat_path = Path("fid_stats/cifar_10_valid_fid_stats.npz")
score = get_fid(Gen, 128, 10000, 256, stat_path)
print(f"\nFID score: {score}")

rc("animation", html="jshtml")
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani