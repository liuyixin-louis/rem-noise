from PIL import Image
import numpy as np
import torch
import cv2
import torchvision
import os
import pickle
import torchvision.transforms as transforms
from .shortcut import shortcut_noise

class ElementWiseTransform():
    def __init__(self, trans=None):
        self.trans = trans

    def __call__(self, x):
        if self.trans is None: return x
        return torch.cat( [self.trans( xx.view(1, *xx.shape) ) for xx in x] )


class Dataset(torchvision.datasets.DatasetFolder):
    def __init__(self, dataset):
        assert isinstance(dataset, torchvision.datasets.DatasetFolder)
        self.loader    = dataset.loader
        self.classes   = dataset.classes
        self.samples   = dataset.samples
        self.y         = dataset.targets
        self.transform = dataset.transform
        self.target_transform = None

        print("numpy.unique", np.unique(self.y))

    def __getitem__(self, idx):
        x,y = super().__getitem__(idx)
        if self.classes == 2:
            y = 0 if y == 16 else 1
        return x,y

    def __len__(self):
        return len(self.y)


class IndexedTensorDataset(Dataset):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.loader    = dataset.loader
        self.classes   = dataset.classes
        self.samples   = dataset.samples
        self.y         = dataset.y
        self.transform = transforms.Compose([ transforms.Resize( [256, 256] ) ])
        self.target_transform = None

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        ''' transform HWC pic to CWH pic '''
        x = np.asarray(x, dtype=np.uint8)
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1)
        return x, y, idx

    def __len__(self):
        return len(self.y)


class PoisonedDataset(Dataset):
    def __init__(self, dataset, noise, fitr=None):
        assert isinstance(dataset, Dataset)
        self.loader    = dataset.loader
        self.classes   = dataset.classes
        self.samples   = dataset.samples
        self.y         = dataset.y
        self.transform = transforms.Compose([ transforms.Resize([ noise.shape[1], noise.shape[2] ]) ])
        self.target_transform = None
        self.data_transform = dataset.transform
        self.data_fitr      = fitr
        ''' the shape of the noise should be (NHWC) '''
        self.noise     = noise

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = (np.asarray(x, dtype=np.int16) + self.noise[idx].astype(np.int16)).clip(0, 255).astype(np.uint8)

        ''' low pass filtering '''
        if self.data_fitr is not None:
            x = self.data_fitr(x)

        x = self.data_transform( Image.fromarray(x) )
        # x = self.data_transform( Image.fromarray(x.astype(np.uint8)) )
        return x, y

    def __len__(self):
        return len(self.y)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.loader    = dataset.loader
        self.classes   = dataset.classes
        self.samples   = dataset.samples
        self.y         = dataset.y
        self.transform = dataset.transform
        self.target_transform = None

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, idx

    def __len__(self):
        return len(self.y)


class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=4):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples


def datasetImageNet(root='./data', train=True, transform=None):
    if train: root = os.path.join(root, 'train')
    else: root = os.path.join(root, 'val')
    return torchvision.datasets.ImageFolder(root=root, transform=transform)

def datasetImageNetKClass(root='./data', train=True, transform=None, k=3):
    dataset = datasetImageNet(root=root, train=train, transform=transform)
    ''' imagenet-ten-class is a subset of the k classes of ImageNet '''
    idx = np.where( np.array(dataset.targets) < k )[0]
    dataset.samples = [ dataset.samples[ii] for ii in idx ]
    dataset.targets = [ dataset.targets[ii] for ii in idx ]
    print("samples number", len(dataset.samples))
    dataset.classes=k
    return dataset

def datasetImageNetTenClass(root='./data', train=True, transform=None):
    dataset = datasetImageNet(root=root, train=train, transform=transform)
    ''' imagenet-ten-class is a subset of the 10 classes of ImageNet '''
    idx = np.where( np.array(dataset.targets) < 10 )[0]
    dataset.samples = [ dataset.samples[ii] for ii in idx ]
    dataset.targets = [ dataset.targets[ii] for ii in idx ]
    dataset.classes=10
    return dataset

def datasetImageNetTwoClass(root='./data', train=True, transform=None):
    dataset = datasetImageNet(root=root, train=train, transform=transform)
    ''' imagenet-two-class is a subset of the 2 classes of ImageNet '''
    idx = np.where((np.array(dataset.targets) == 16)|(np.array(dataset.targets) == 107))[0]
    dataset.samples = [ dataset.samples[ii] for ii in idx ]
    dataset.targets = [ dataset.targets[ii] for ii in idx ]
    dataset.targets = [ 0 if y == 16 else 1 for y in dataset.targets ]
    print(f'imagenet-two-class {"train" if train else "test"} dataset size:', len(dataset))
    dataset.classes=2
    return dataset

def datasetImageNetMini(root='./data', train=True, transform=None):
    dataset = datasetImageNet(root=root, train=train, transform=transform)
    ''' imagenet-mini is a subset of the first 100 classes of ImageNet '''
    idx = np.where( np.array(dataset.targets) < 100 )[0]
    dataset.samples = [ dataset.samples[ii] for ii in idx ]
    dataset.targets = [ dataset.targets[ii] for ii in idx ]
    return dataset


def get_transforms(dataset, train=True, is_tensor=True,close_trans=False):
    assert ("imagenet" in dataset)

    if train and close_trans is False:
        comp1 = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), ]
    else:
        comp1 = [
            transforms.Resize( [256, 256] ),
            transforms.CenterCrop(224), ]
        
    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = ElementWiseTransform(trans)

    return trans


def get_filter(fitr):
    if fitr == 'averaging':
        return lambda x: cv2.blur(x, (3,3))
    elif fitr == 'gaussian':
        return lambda x: cv2.GaussianBlur(x, (3,3), 0)
    elif fitr == 'median':
        return lambda x: cv2.medianBlur(x, 3)
    elif fitr == 'bilateral':
        return lambda x: cv2.bilateralFilter(x, 9, 75, 75)

    raise ValueError


def get_dataset(dataset, root='./data', train=True,args=None):
    assert ("imagenet" in dataset)

    transform = get_transforms(dataset, train=train, is_tensor=False,close_trans=args.close_trans if args is not None else None)

    if dataset == 'imagenet':
        target_set = datasetImageNet(root=root, train=train, transform=transform)
    elif dataset == 'imagenet-mini':
        target_set = datasetImageNetMini(root=root, train=train, transform=transform)
    elif dataset == "imagenet-two-class":
        # target_set = datasetImageNetTwoClass(root=root, train=train, transform=transform)
        target_set = datasetImageNetKClass(root=root, train=train, transform=transform, k= 2)
    elif dataset == "imagenet-ten-class":
        # target_set = datasetImageNetTenClass(root=root, train=train, transform=transform)
        target_set = datasetImageNetKClass(root=root, train=train, transform=transform, k= 10)
    elif dataset == "imagenet-three-class":
        target_set = datasetImageNetKClass(root=root, train=train, transform=transform, k= 3)

    target_set = Dataset(target_set)

    return target_set


def get_indexed_loader(dataset, batch_size, root='./data', train=True,args=None):
    target_set = get_dataset(dataset, root=root, train=train,args=args)

    if train:
        target_set = IndexedDataset(target_set)
    else:
        pass

    if train:
        loader = Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_indexed_tensor_loader(dataset, batch_size, root='./data', train=True,args=None):
    target_set = get_dataset(dataset, root=root, train=train, args=args)
    target_set = IndexedTensorDataset(target_set)

    if train:
        loader = Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_loader_shortcut(
        dataset, batch_size, root='./data', train=True, noise_rate=1.0, fitr=None,args=None):

    target_set = get_dataset(dataset, root=root, train=train, args=args)

    # if noise_rate < 1.0:
    #     raise NotImplementedError

    
    print("type of target_set sample:", type(target_set.samples))
    print("len of target_set sample:", len(target_set.samples))
    raw_noise = shortcut_noise(target_set.samples, target_set.y, noise_frame_size=8, norm_ball=args.defense_budget, img_size=224, c=3).astype(np.int16)
    
    print("shortcut noise generating....")
    import time
    t1 = time.time()
    noise = np.zeros_like(raw_noise)
        # if poisoned_indices_path is not None:
        #     with open(poisoned_indices_path, 'rb') as f:
        #         indices = pickle.load(f)
        # else:
    t2 = time.time()
    print("shortcut noise finish generate!")
    print("cost second:", t2-t1, " s")
    indices = np.random.permutation(len(noise))[:int(len(noise)*noise_rate)]

    noise[indices] += raw_noise[indices]

    ''' restore noise (NWHC) for raw images (NHWC) '''
    noise = np.transpose(noise, [0,2,1,3])

    lp_fitr = None if fitr is None else get_filter(fitr)

    target_set = PoisonedDataset(target_set, noise, fitr=lp_fitr)

    if train:
        loader = Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader

def get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0, poisoned_indices_path=None, fitr=None,args=None):

    target_set = get_dataset(dataset, root=root, train=train,args=args)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        # raw_noise = raw_noise.astype(np.int32)

        if noise_rate < 1.0:
            raise NotImplementedError

        noise = raw_noise

        ''' restore noise (NCWH) for raw images (NWHC) '''
        noise = np.transpose(noise, [0,2,3,1])

        lp_fitr = None if fitr is None else get_filter(fitr)

        target_set = PoisonedDataset(target_set, noise, fitr=lp_fitr)

    else:
        pass

    if train:
        loader = Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_clear_loader(
        dataset, batch_size, root='./data', train=True,
        noise_rate=1.0, poisoned_indices_path=None, args=None):

    raise NotImplementedError
