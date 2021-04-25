# Modified from 
# 1) https://github.com/w86763777/pytorch-inception-score-fid/blob/master/score/both.py

import math

from tqdm.auto import tqdm
import numpy as np
import torch

from metrics.torch_inception import InceptionV3
from metrics.torch_fid_utils import calculate_frechet_distance, torch_cov
from utils.utils import check_gpu

device = check_gpu()
torch.backends.cudnn.deterministic = True

def is_fid_from_generator(generator, 
                          latent_dims,
                          num_classes, 
                          num_imgs,
                          batch_sz,
                          fid_stat_path):
    # Randomly generate the same set of z vectors each time
    torch.manual_seed(42)
    generator.eval()
    with torch.no_grad():
        eval_iter = math.ceil(num_imgs / batch_sz)
        img_list = []
        for i in tqdm(range(eval_iter), leave=False, desc='generating images'):
            b_sz = min(batch_sz, num_imgs-i*batch_sz)
            z = torch.randn(b_sz, latent_dims, device=device)
            fake_labels = torch.randint(high=num_classes, size=(b_sz,),
                             device=device)
            gen_imgs = generator(z, class_idx = fake_labels)
            if isinstance(gen_imgs, tuple):
                gen_imgs = gen_imgs[0]
            img_list += [gen_imgs]

        img_list = torch.cat(img_list, 0)
        is_score, fid_score = calc_is_and_fid(img_list,
                                              fid_stat_path,
                                              batch_size=max(batch_sz, 128))
        return is_score, fid_score
    

def calc_is_and_fid(images,
                    fid_stats_path,
                    splits=10,
                    batch_size=50,
                    in_zero_one=False,
                    verbose=True):
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.
    Args:
        images: tensor of shape (num_images x 3 x H x W). 
                Expected Range [0,1] or [-1,1] with `in_zero_one` set.
        fid_stats_path: str, Path to pre-calculated statistic
        splits: The number of bins of Inception Score. Default is 10.
        batch_size: int, The batch size for calculating activations.
        in_zero_one: if `images` are in range [0,1] 
        verbose: int. Set verbose to False for disabling progress bar. Otherwise,
                 the progress bar is showing when calculating activations.
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    if not in_zero_one:
        images = (images + 1.0)/2.0
    num_images = len(images)
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    fid_acts = torch.empty((num_images, 2048)).to(device)
    is_probs = torch.empty((num_images, 1008)).to(device)

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="inception_score_and_fid")
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        batch_images = images[start: start + batch_size]
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            fid_acts[start: end] = pred[0].view(-1, 2048)
            is_probs[start: end] = pred[1]

        start = end
        pbar.update(len(batch_images))
    pbar.close()
    
    # Inception Score
    scores = []
    for i in range(splits):
        part = is_probs[
            (i * is_probs.shape[0] // splits):
            ((i + 1) * is_probs.shape[0] // splits), :]
        kl = part * (
            torch.log(part) -
            torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        scores.append(torch.exp(kl))
    
    scores = torch.stack(scores)
    is_score = (torch.mean(scores).cpu().item(),
                torch.std(scores).cpu().item())

    # FID Score
    f = np.load(fid_stats_path)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    m1 = torch.mean(fid_acts, axis=0)
    s1 = torch_cov(fid_acts, rowvar=False)
    m2 = torch.as_tensor(m2).to(m1.dtype).to(device)
    s2 = torch.as_tensor(s2).to(s1.dtype).to(device)

    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score.cpu().item()
