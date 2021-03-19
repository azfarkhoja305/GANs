from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class ImageDataset:
    def __init__(self, dataset = 'cifar_10', batch_sz=256, tfms=None, resize_img=None, num_workers=2):
        if tfms is None:
            tfms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            if resize_img is not None:
                tfms.append(transforms.Resize((resize_img, resize_img)))
        if dataset.lower() == 'cifar_10':
            train_dataset = datasets.CIFAR10(root='data/cifar_10', train=True,
                                             transform=transforms.Compose(tfms), download=True)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, 
                                           num_workers=num_workers)
            valid_dataset = datasets.CIFAR10(root='data/cifar_10', train=False, 
                                             transform=transforms.Compose(tfms))
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=False, 
                                           num_workers=num_workers)
        else:
            raise NotImplementedError(f'Unkown dataset: {dataset}')