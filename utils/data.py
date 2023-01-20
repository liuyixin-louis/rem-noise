from PIL import Image
import numpy as np
import torch
import torchvision
import os
import pickle


class ElementWiseTransform():
    def __init__(self, trans=None):
        self.trans = trans

    def __call__(self, x):
        if self.trans is None: return x
        return torch.cat( [self.trans( xx.view(1, *xx.shape) ) for xx in x] )

    def __pair_trans__(self, x, y):
        if self.trans is None: return x,y
        else:
            # all_concat = []
            assert x.shape[0] == y.shape[0] == 128
            # torch.cat([self.trans(xx.view(1, *xx.shape)) for xx in x])
            all_concat = torch.cat([self.trans(torch.cat([x[i].view(1,*x[i].shape),y[i].view(1,*y[i].shape)])) for i in range(x.shape[0])])
            return all_concat[0::2], all_concat[1::2]

class IndexedTensorDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        ''' transform HWC pic to CWH pic '''
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1)
        return x, y, idx

    def __len__(self):
        return len(self.x)


class Dataset():
    def __init__(self, x, y, transform=None, fitr=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.fitr = fitr

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        ''' low pass filtering '''
        if self.fitr is not None:
            x = self.fitr(x)

        ''' data augmentation '''
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )

        return x, y

    def __len__(self):
        return len(self.x)


class IndexedDataset():
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.ii = np.array( range(len(x)), dtype=np.int64 )
        self.transform = transform

    def __getitem__(self, idx):
        x, y, ii = Image.fromarray(self.x[idx]), self.y[idx], self.ii[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, ii

    def __len__(self):
        return len(self.x)


def datasetCIFAR10(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def datasetCIFAR100(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)

def datasetTinyImageNet(root='./path', train=True, transform=None):
    if train: root = os.path.join(root, 'tiny-imagenet_train.pkl')
    else: root = os.path.join(root, 'tiny-imagenet_val.pkl')
    with open(root, 'rb') as f:
        dat = pickle.load(f)
    return Dataset(dat['data'], dat['targets'], transform)
    # root = os.path.join(root, 'tiny-imagenet-200')
    # if train: root = os.path.join(root, 'train')
    # else: root = os.path.join(root, 'val', 'images')
    # raw_dataset = torchvision.datasets.ImageFolder(root)
    # xx, yy = [], []
    # for i in range( len(raw_dataset) ):
    #     x, y = raw_dataset[i]
    #     x = np.array(x)
    #     xx.append( x.reshape(1, *x.shape) )
    #     yy.append( y )
    # xx = np.concatenate(xx)
    # yy = np.array(yy)

    # dat = {'data':xx, 'targets':yy}
    # if train: save_name = 'tiny-imagenet_train.pkl'
    # else: save_name = 'tiny-imagenet_val.pkl'

    # import pickle
    # with open('./data/{}'.format(save_name), 'wb') as f:
    #     pickle.dump(dat, f)
    # exit()
    # return Dataset(xx, yy, transform)


class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False ,args=None):
        if args is not None:
            num_workers = args.num_workers
        else:
            num_workers=4
        print('num_workers:',num_workers)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                  drop_last=drop_last, num_workers=num_workers, pin_memory=True)
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
