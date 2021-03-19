import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def check_gpu():
    if torch.cuda.is_available():    return torch.device('cuda')
    return torch.device('cpu')

def display_images(image_holder, ax=None, title=None, max_idx=128):
    """ Randomly plots images from 'image_holder' """
    if ax is None:
         _, ax = plt.subplots(figsize=(9,9))
    ax.set_axis_off()
    indices = np.random.randint(low = 0, high = max_idx, size = (64,))
    if isinstance(image_holder, DataLoader):
        real_batch = next(iter(image_holder))[0]
        if title is None:    ax.set_title("Real Images")
        else:    ax.set_title(title)
        ax.imshow(np.transpose(vutils.make_grid(real_batch[indices], padding=2, normalize=True),(1,2,0)))
    else: 
        if isinstance(image_holder, list):  
            image_holder = image_holder[-1]
        elif isinstance(image_holder, (torch.Tensor, np.ndarray)):
            if isinstance(image_holder, np.ndarray):
                image_holder = torch.as_tensor(image_holder)
            assert image_holder.ndim == 4, f'image_holder needs 4 dims, has {image_holder.ndim}'
            image_holder = vutils.make_grid(image_holder[indices], padding=2, normalize=True)
        else:    raise Exception(f'type: {type(image_holder)} for image_holder not accepted')
        if title is None:    ax.set_title("Fake Images")
        else:    ax.set_title(title)
        ax.imshow(np.transpose(image_holder,(1,2,0)))