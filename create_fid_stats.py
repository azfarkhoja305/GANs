import argparse
from pathlib import Path
import pdb

import numpy as np
import torch


def make_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, 
                        help='dataset name')
    parser.add_argument('-t', '--train', required=True, type=str, 
                        help='Set True if train set')
    parser.add_argument('-s','--save', required=False, type=str, 
                        default='fid_stats', help='directory for saving stats')
    return parser.parse_args()

def main(args):
    device = check_gpu()
    dataset =  ImageDataset(args.dataset, batch_sz = 256)

    if args.train == 'True':
        t_or_v = 'train'
        loader = dataset.train_loader
    elif args.train == 'False':
        t_or_v='valid'
        loader = dataset.valid_loader
    else:    raise Exception('args.train not understood')

    all_images = [imgs for imgs,_ in loader]
    all_images = torch.cat(all_images, 0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    # len(all_images) needs to be divisble by batch_size to consider entire dataset
    # TODO: FIX calculate_activation_statistics
    bs = 50 if str(device) == 'cpu' else 100
    mu, sigma = calculate_activation_statistics(all_images, model, batch_size=bs)
    mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()

    save_path = Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez(save_path/f'{args.dataset}_{t_or_v}_fid_stats')


if __name__=='__main__':
    from utils.torch_fid_score import calculate_activation_statistics
    from utils.inception import InceptionV3
    from utils.utils import check_gpu
    from datasets import ImageDataset

    args = make_argparse()
    print(args)
    main(args)
