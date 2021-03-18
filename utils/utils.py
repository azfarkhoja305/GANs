import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def check_gpu():
    if torch.cuda.is_available():   return torch.device('cuda')
    return torch.device('cpu')

def display_images(image_holder, ax=None, high_idx=128):
    """ Randomly plots images from 'image_holder' """
    if ax is None:
         _, ax = plt.subplots(figsize=(9,9))
    ax.set_axis_off()
    indices = np.random.randint(low = 0, high = high_idx,size = (64,))
    if isinstance(image_holder, DataLoader):
        real_batch = next(iter(image_holder))[0]
        ax.set_title("Real Images")
        ax.imshow(np.transpose(vutils.make_grid(real_batch[indices], nrow=8, padding=2, normalize=True),(1,2,0)))
    else: 
        if isinstance(image_holder, list):  
            image_holder = image_holder[-1]
        if isinstance(image_holder, torch.Tensor):
            assert image_holder.ndim == 4
            image_holder = vutils.make_grid(image_holder[indices], padding=2, normalize=True)
        ax.set_title("Fake Images")
        ax.imshow(np.transpose(image_holder,(1,2,0)))