import random
import re
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def check_gpu():
    if torch.cuda.is_available():    return torch.device('cuda')
    return torch.device('cpu')

def set_seed(seed=123):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def reduce_resolution(image_batch):
    """ Function to reduce the resolution of an image by half. """
    return F.interpolate(image_batch, scale_factor=0.5, mode='bilinear',
                          align_corners=False, recompute_scale_factor=False)

class AvgLossMetric():
    """ Average loss statistics over an entire epoch """
    def __init__(self):
        self.loss = 0
        self.num = 0
    def update_state(self,loss,num):
        assert num > 0, f'num elements = {num}, needs to be greater than zero'
        # since incoming loss is averaged
        self.loss += loss * num
        self.num +=num
    def result(self):
        assert self.num > 0, f'num = {self.num}, needs to be greater than zero'
        # avg loss for one epoch = total loss / total number of samples
        return self.loss / self.num
    def reset_state(self):
        self.loss = 0
        self.num = 0


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


class Checkpoint():
    """ Saves checkpoints at required epochs. Additionally 
        automatically picks up the latest checkpoint if the folder already exists.
        Can also load the checkpoint given the file """
    def __init__(self, ckp_folder, max_epochs, num_ckps, start_after=0.5):
        """ Start checkpointing after `start_after*max_epoch`. 
            Like start after 50% of max_epochs completed and divides the number of
            checkpoints equally. """
        self.ckp_folder = ckp_folder
        self.max_epochs = max_epochs
        self.num_ckps = num_ckps
        self.ckp_epochs = np.linspace(start_after*max_epochs, max_epochs, 
                                      num_ckps, dtype=np.int).tolist() 
        if isinstance(self.ckp_folder, str):
            self.ckp_folder = Path(self.ckp_folder)
    
    def check_if_exists(self, generator, critic, gen_opt, critic_opt ):
        if not self.ckp_folder.exists():
            self.ckp_folder.mkdir(parents=True)
            return generator, critic, gen_opt, critic_opt, 0, None

        ckp_files = [file for file in self.ckp_folder.ls() if file.suffix in ['.pth','.pt']]
        if not ckp_files:
            return  generator, critic, gen_opt, critic_opt, 0, None
        print("Checkpoint folder with checkpoints already exists. Searching for the latest.")
        # finding latest (NOT best) checkpoint to resume train
        numbers = [int(re.search(r'\d+', name.stem).group()) for name in ckp_files]
        idx = max(enumerate(numbers), key=lambda x: x[1])[0]
        return self.load_checkpoint(ckp_files[idx], generator, critic, gen_opt, critic_opt)

    def at_epoch_end(self, generator, critic, gen_opt, critic_opt, epoch, loss_logs):
        if epoch in self.ckp_epochs:
            self.save_checkpoint(self.ckp_folder/f'GanModel_{epoch}.pth',
                                 generator, critic, gen_opt, critic_opt,
                                 epoch, loss_logs)

    @staticmethod
    def load_checkpoint(ckp_path, generator, critic, gen_opt=None, critic_opt=None):
        assert isinstance(generator, nn.Module), f'Generator is not nn.Module'
        assert isinstance(critic, nn.Module), f'Discriminator is not nn.Module'
        if isinstance(ckp_path, str): 
            ckp_path = Path(ckp_path)
        assert ckp_path.exists(), f'Checkpoint File: {str(ckp_path)} does not exist'
        print(f"=> Loading checkpoint: {ckp_path}")
        ckp = torch.load(ckp_path)
        generator.load_state_dict(ckp['generator_state_dict'])
        critic.load_state_dict(ckp['critic_state_dict'])
        if gen_opt is not None and ckp['gen_optim_state_dict'] is not None:
            gen_opt.load_state_dict(ckp['gen_optim_state_dict'])
        if critic_opt is not None and ckp['critic_optim_state_dict'] is not None:
            critic_opt.load_state_dict(ckp['critic_optim_state_dict'])

        epoch_complete = ckp['epoch']
        loss_logs = ckp['loss_logs']
        return generator, critic, gen_opt, critic_opt, epoch_complete+1, loss_logs
    
    @staticmethod
    def save_checkpoint(file_path, generator, critic, gen_opt=None, 
                        critic_opt=None, epoch=-1, loss_logs=None):
        assert not file_path.is_dir(), f"`file_path` cannot be a dir, Needs to be dir/file_name"
        ckp_suffix = ['.pth','.pt']
        assert file_path.suffix in ckp_suffix, f'{file_path.name} is not in checkpoint file format'
        assert isinstance(generator, nn.Module), f'Generator is not nn.Module'
        assert isinstance(critic, nn.Module), f'Discriminator is not nn.Module'
        print(f"=> Saving Checkpoint with name `{file_path.name}`")
        gen_opt_dict = gen_opt.state_dict() if gen_opt is not None else None
        critic_opt_dict = critic_opt.state_dict() if critic_opt is not None else None
        torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'gen_optim_state_dict':  gen_opt_dict,
                    'critic_optim_state_dict': critic_opt_dict,
                    'epoch': epoch,
                    'loss_logs': loss_logs
                    }, file_path)

    @staticmethod
    def delete_checkpoint(file_path):
        if isinstance(file_path, str): 
            file_path = Path(file_path)
        ckp_suffix = ['.pth','.pt']
        assert file_path.suffix in ckp_suffix, f'{file_path.name} is not in checkpoint file format'
        assert file_path.exists(), f"`file_path`: {str(file_path)} not found" 
        print(f"Deleting {str(file_path)}")
        file_path.unlink()
    
    def find_best_ckp():
        """ Calculate the metric for each checkpoint and return best"""
        raise NotImplementedError