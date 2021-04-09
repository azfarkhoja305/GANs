# Implementation from
# https://github.com/VITA-Group/TransGAN/blob/7e5fa2d0c4d45ed2bf89068f0a9edb61a2a6db33/functions.py

import torch
import numpy as np


def wgangp_eps_loss(
    dis_net, real_imgs, fake_imgs, phi, real_validity, fake_validity, use_cpu=False
):
    gradient_penalty = compute_gradient_penalty(
        dis_net, real_imgs, fake_imgs.detach(), phi, use_cpu
    )
    d_loss = (
        -torch.mean(real_validity)
        + torch.mean(fake_validity)
        + gradient_penalty * 10 / (phi ** 2)
    )
    d_loss += (torch.mean(real_validity) ** 2) * 1e-3
    return d_loss


def compute_gradient_penalty(D, real_samples, fake_samples, phi, use_cpu):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if use_cpu:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    else:
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
            real_samples.get_device()
        )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    if use_cpu:
        fake = torch.ones([real_samples.shape[0], 1], requires_grad=False)
    else:
        fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(
            real_samples.get_device()
        )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty
