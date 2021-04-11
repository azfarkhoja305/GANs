import argparse
from pathlib import Path

import numpy as np


def make_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, 
                        help='dataset name')
    parser.add_argument('-t', '--train', required=True, type=str, 
                        help='Set True if train set')
    parser.add_argument('-b','--batch_size', required=False, type=int,
                        default=256, help='Batch Size')
    parser.add_argument('-s','--save', required=False, type=str, 
                        default='fid_stats', help='directory for saving stats')
    return parser.parse_args()

def main(args):
    device = check_gpu()
    bs = 50 if str(device) == 'cpu' else args.batch_size
    dataset =  ImageDataset(args.dataset, batch_sz = bs)
    if args.train.lower() == 'true':
        t_or_v = 'train'
        loader = dataset.train_loader
    elif args.train.lower() == 'false':
        t_or_v='valid'
        loader = dataset.valid_loader
    else:    raise Exception('args.train not understood')

    save_path = Path(args.save)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{t_or_v}_fid_stats'
    if (save_path/(file_name + '.npz')).exists():
        print(f"{(save_path/(file_name + '.npz'))} exists. Exiting !!!")
        return

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    mu, sigma = calc_activation_stats(loader, model)
    mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()

    np.savez(save_path/file_name, mu=mu, sigma=sigma)


if __name__=='__main__':
    from metrics.torch_fid_utils import calc_activation_stats
    from metrics.torch_inception import InceptionV3
    from utils.utils import check_gpu, set_seed
    from utils.datasets import ImageDataset

    set_seed(42)

    args = make_argparse()
    print(args)
    main(args)
