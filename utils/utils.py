import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def check_gpu():
    if torch.cuda.is_available():    return torch.device('cuda')
    return torch.device('cpu')

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
         _, ax = plt.subplots(figsize=(9,9))
    ax.set_axis_off()
    if isinstance(image_holder, DataLoader):
        real_batch = next(iter(image_holder))[0]
        batch_sz = len(real_batch)
        indices = np.random.permutation(batch_sz)[:min(64, batch_sz)]
        if title is None:    ax.set_title("Real Images")
        else:    ax.set_title(title)
        ax.imshow(np.transpose(vutils.make_grid(real_batch[indices], nrow=nrow,
                                 padding=2, normalize=True),(1,2,0)))
    else: 
        if isinstance(image_holder, list):  
            image_holder = image_holder[-1]
        elif isinstance(image_holder, (torch.Tensor, np.ndarray)):
            if isinstance(image_holder, np.ndarray):
                image_holder = torch.as_tensor(image_holder)
            assert image_holder.ndim == 4, f'image_holder needs 4 dims, has {image_holder.ndim}'
            batch_sz = len(image_holder)
            indices = np.random.permutation(batch_sz)[:min(64, batch_sz)]
            image_holder = vutils.make_grid(image_holder[indices], nrow=nrow,
                                             padding=2, normalize=True)
        else:    raise Exception(f'type: {type(image_holder)} for image_holder not accepted')
        if title is None:    ax.set_title("Fake Images")
        else:    ax.set_title(title)
        ax.imshow(np.transpose(image_holder,(1,2,0)))