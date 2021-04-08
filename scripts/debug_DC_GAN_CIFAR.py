from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import torch
from torch import nn
from torch import optim
from datasets import ImageDataset
import torchvision.utils as vutils
from torchsummary import summary
from utils.utils import check_gpu, display_images
from models.transformer_generator import TGenerator

# from models.discriminator import Discriminator
from models.ViT_discriminator import Discriminator
from types import SimpleNamespace

from utils.loss import wgangp_eps_loss

device = check_gpu()
print(f"Using device: {device}")

batch_sz = 64
dataset = ImageDataset("cifar_10", batch_sz=batch_sz)

# display_images(dataset.train_loader, max_idx=256)

# Gen = Generator().to(device)
latent_dims = 1024
Gen = TGenerator(latent_dims=latent_dims).to(device)
summary(Gen, (latent_dims,))

# Dis = Discriminator().to(device)
args = SimpleNamespace(**{"d_depth": 7, "df_dim": 384, "img_size": 32, "patch_size": 8})
Dis = Discriminator(args).to(device)
summary(Dis, (3, 32, 32))

loss_fn = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(batch_sz, latent_dims, device=device)
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


for epoch in range(num_epochs):
    for i, data in enumerate(dataset.train_loader):

        ## Train with all-real batch
        Dis.zero_grad()
        # Format batch
        real = data[0].to(device)
        b_size = real.size(0)
        real_validity = torch.full(
            (b_size,), real_label, dtype=torch.float, device=device
        )
        # Forward pass real batch through D
        output_real = Dis(real).view(-1)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_dims, device=device)
        # Generate fake image batch with G
        fake = Gen(noise)
        fake_validity = torch.full_like(real_validity, fake_label)
        # Classify all fake batch with D
        output_fake = Dis(fake.detach()).view(-1)

        errD = wgangp_eps_loss(Dis, real, fake, 1.0, output_real, output_fake)

        errD.backward()

        torch.nn.utils.clip_grad_norm_(Dis.parameters(), 5.0)

        # Update D
        optD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Gen.zero_grad()
        label = torch.full_like(
            real_validity, real_label
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

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # Output training stats
        # if (i+1) %100 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
        #         % (epoch, num_epochs, i, len(dataset.train_loader),
        #              errD.item(), errG.item()))

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