import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import animation, rc
import torch
import torch.optim as optim

from types import SimpleNamespace

Path.ls = lambda x: list(x.iterdir())
if Path("./GANs").exists():
    sys.path.insert(0, "./GANs")

from models.transformer_generator import TGenerator
from models.ViT_discriminator import Discriminator
from utils.utils import check_gpu, display_images
from utils.checkpoint import Checkpoint
from utils.loss import wgangp_eps_loss
from utils.datasets import ImageDataset
from metrics.torch_is_fid_score import is_fid_from_generator

gdrive = "/mnt/c/Google Drive/"

# Create a required checkpoint instance.
# If does not exists, Checkpoint class will create one.
ckp_folder = gdrive + "temporary_checkpoint"

device = check_gpu()
print(f"Using device: {device}")

"""# Training"""

gen_batch_sz = 64
dis_batch_sz = 32
latent_dims = 1024
lr, beta1, beta2 = 1e-4, 0, 0.999
num_epochs = 20

dataset = ImageDataset("cifar_10", batch_sz=dis_batch_sz, num_workers=2)
# display_images(dataset.train_loader)

Gen = TGenerator(latent_dims=latent_dims).to(device)
fixed_z = torch.randn(gen_batch_sz, latent_dims, device=device)
# summary(Gen,(latent_dims,))

args = SimpleNamespace(**{"d_depth": 7, "df_dim": 384, "img_size": 32, "patch_size": 8})
Dis = Discriminator(args).to(device)
# summary(Dis,(3,32,32,))

optG = optim.AdamW(Gen.parameters(), lr=lr, betas=(beta1, beta2))
optD = optim.AdamW(Dis.parameters(), lr=lr, betas=(beta1, beta2))

img_list = []
G_losses = []
D_losses = []
loss_logs = {"gen_loss": [], "dis_loss": []}
iters = 0

ckp_class = Checkpoint(ckp_folder, max_epochs=num_epochs, num_ckps=5, start_after=0.1)

# check if any existing checkpoint exists, none found hence start_epoch is 0.
# Optimizer states also get saved
Gen, Dis, optG, optD, start_epoch, old_logs = ckp_class.check_if_exists(
    Gen, Dis, optG, optD
)

loss_logs = old_logs or loss_logs
print(start_epoch)  # , loss_logs

# Commented out IPython magic to ensure Python compatibility.
for epoch in range(start_epoch, num_epochs + 1):
    for i, data in enumerate(dataset.train_loader):

        ###########################
        # (1) Update Dis network
        ###########################

        ## Train with all-real batch
        Dis.zero_grad()
        real = data[0].to(device)
        output_real = Dis(real).view(-1)

        ## Train with all-fake batch
        dis_z = torch.randn(dis_batch_sz, latent_dims, device=device)
        fake_1 = Gen(dis_z).detach()
        output_fake_1 = Dis(fake_1).view(-1)

        ## Compute loss and backpropagate
        errD = wgangp_eps_loss(
            Dis, real, fake_1, 1.0, output_real, output_fake_1, use_cpu=True
        )
        errD.backward()
        torch.nn.utils.clip_grad_norm_(Dis.parameters(), 5.0)
        optD.step()

        ###########################
        # (2) Update Gen network
        ###########################

        Gen.zero_grad()
        gen_z = torch.randn(gen_batch_sz, latent_dims, device=device)
        fake_2 = Gen(gen_z)
        output_fake_2 = Dis(fake_2).view(-1)
        errG = -torch.mean(output_fake_2)
        errG.backward()
        torch.nn.utils.clip_grad_norm_(Gen.parameters(), 5.0)
        optG.step()

        ###########################
        # (3) Output
        ###########################

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

    loss_logs["gen_loss"].append(errG.item())  # TODO: mean loss per epoch
    loss_logs["dis_loss"].append(errD.item())

    # Checkpoint
    ckp_class.at_epoch_end(Gen, Dis, optG, optD, epoch=epoch, loss_logs=loss_logs)

"""# Analysis"""

_, axs = plt.subplots(1, 2, figsize=(15, 15))
display_images(dataset.train_loader, ax=axs[0])
display_images(img_list, ax=axs[1])
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.ylim([-10, 10])
plt.legend()
plt.show()

"""Calculating FID Score"""

stat_path = Path("fid_stats/cifar_10_valid_fid_stats.npz")
inception_score, fid = is_fid_from_generator(
    generator=Gen,
    latent_dims=latent_dims,
    num_imgs=10000,
    batch_sz=256,
    fid_stat_path=stat_path,
)

print(f"\nFID score: {fid}")
print(f"\nIS: {inception_score}")