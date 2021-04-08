from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ImageDataset:
    def __init__(self, dataset = 'cifar_10', batch_sz=256, tfms=None, 
                resize=None, num_workers=2):

        """ Loads the dataset with transforms and sets both train and valid dataloader.
        
            tfms: List of extra transforms for the train set.
            Note: If image need to be resized, `resize` is required for valid transforms.
            Preferable to not add transforms.Resize(resize) in tfms, altough there 
            is a check to prevent it twice in train transforms.
        """
        
        valid_tfms = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]

        if tfms is None:
            tfms = valid_tfms
        else:    tfms.extend(valid_tfms)

        if resize is not None:
            valid_tfms.append(transforms.Resize(resize))
            if not any(isinstance(t, transforms.Resize) for t in tfms):
                tfms.append(transforms.Resize(resize))

        self.train_transforms = tfms
        self.valid_transforms = valid_tfms

        if dataset.lower() == 'cifar_10':
            train_dataset = datasets.CIFAR10(root='data/cifar_10', train=True,
                                             transform=transforms.Compose(tfms), download=True)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, 
                                           num_workers=num_workers, drop_last=True)
            valid_dataset = datasets.CIFAR10(root='data/cifar_10', train=False, 
                                             transform=transforms.Compose(valid_tfms))
            self.valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=False, 
                                           num_workers=num_workers, drop_last=True)
        else:
            raise NotImplementedError(f'Unkown dataset: {dataset}')