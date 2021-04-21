import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def check_gpu():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed=123):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reduce_resolution(image_batch):
    """ Function to reduce the resolution of an image by half. """
    return F.interpolate(
        image_batch,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )


def display_images(image_holder, ax=None, title=None, nrow=8):
    """
    Randomly plots images from 'image_holder' or
    fixed images from torchvision.utils.make_grid passed as a list.
    Args:
    image_holder: can be torch Dataloader, batch of image tensor, or a list of
                  torchvision.utils.make_grid. If type Dataloader or tensor randomly
                  samples images.
    ax: matplotlib axis in case of subplots, if None creates one.
    title: Title for the plot.
    nrows: number of images per row, default=8. We fix total_images to min(Batch_Sz,64)
           - Default behaviour of torchvision.utils.make_grid is to
             have (Batch_Sz / nrow, nrow)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    ax.set_axis_off()
    if isinstance(image_holder, DataLoader):
        iterator = iter(image_holder)
        # pick up multiple batches, but show 64 images
        real_batch = torch.cat([next(iterator)[0] for i in range(4)])
        batch_sz = len(real_batch)
        indices = np.random.permutation(batch_sz)[: min(64, batch_sz)]
        if title is None:
            ax.set_title("Real Images")
        else:
            ax.set_title(title)
        ax.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[indices], nrow=nrow, padding=2, normalize=True
                ),
                (1, 2, 0),
            )
        )
    else:
        if isinstance(image_holder, list):
            image_holder = image_holder[-1]
        elif isinstance(image_holder, (torch.Tensor, np.ndarray)):
            if isinstance(image_holder, np.ndarray):
                image_holder = torch.as_tensor(image_holder)
            assert (
                image_holder.ndim == 4
            ), f"image_holder needs 4 dims, has {image_holder.ndim}"
            batch_sz = len(image_holder)
            indices = np.random.permutation(batch_sz)[: min(64, batch_sz)]
            image_holder = vutils.make_grid(
                image_holder[indices], nrow=nrow, padding=2, normalize=True
            )
        else:
            raise Exception(f"type: {type(image_holder)} for image_holder not accepted")
        if title is None:
            ax.set_title("Fake Images")
        else:
            ax.set_title(title)
        ax.imshow(np.transpose(image_holder, (1, 2, 0)))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_uniform_(m.weight.data, 1.)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
