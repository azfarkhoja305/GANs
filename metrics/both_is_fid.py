# Modified from 
# 1) https://github.com/w86763777/pytorch-inception-score-fid/blob/master/score/both.py

import numpy as np
import torch
from tqdm.auto import tqdm

from metrics.inception import InceptionV3
from metrics.fid import calculate_frechet_distance, torch_cov
from utils.utils import check_gpu

device = check_gpu()

def get_inception_score_and_fid(
        images,
        fid_stats_path,
        splits=10,
        batch_size=50,
        verbose=True):
    """Calculate Inception Score and FID.
    For each image, only a forward propagation is required to
    calculating features for FID and Inception Score.
    Args:
        images: List of tensor or torch.utils.data.Dataloader. The return image
                must be float tensor of range [0, 1].
        fid_stats_path: str, Path to pre-calculated statistic
        splits: The number of bins of Inception Score. Default is 10.
        batch_size: int, The batch size for calculating activations. If
                    `images` is torch.utils.data.Dataloader, this arguments
                    does not work.
        use_torch: bool. The default value is False and the backend is same as
                   official implementation, i.e., numpy. If use_torch is
                   enableb, the backend linalg is implemented by torch, the
                   results are not guaranteed to be consistent with numpy, but
                   the speed can be accelerated by GPU.
        verbose: int. Set verbose to 0 for disabling progress bar. Otherwise,
                 the progress bar is showing when calculating activations.
    Returns:
        inception_score: float tuple, (mean, std)
        fid: float
    """
    num_images = len(images)
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    fid_acts = torch.empty((num_images, 2048)).to(device)
    is_probs = torch.empty((num_images, 1008)).to(device)

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_score_and_fid")
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
    m2 = torch.tensor(m2).to(m1.dtype).to(device)
    s2 = torch.tensor(s2).to(s1.dtype).to(device)

    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score