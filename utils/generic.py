import pickle
import os
import sys
import logging
import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from .shortcut import shortcut_noise
import models
from . import data
from . import imagenet_utils
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def add_shared_args(parser):
#     parser.add_argument('--state', type=str, default='eval', help='which state this exp currently in', choice=['train', 'eval'])
    
#     return parser
    

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.
    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True,close_trans=False):
    if dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenet-two-class':
        return imagenet_utils.get_transforms(dataset, train, is_tensor,close_trans)

    if train and close_trans == False:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        elif "imagenet" in dataset:
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 8), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []
        print(f'trian is {train}')
        print('comp1 is  []')

    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

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

def random_dic(dicts):
    import random
    dict_key_ls = [(dicts['image'][i],dicts['label'][i]) for i in range(len(dicts['image']))]
    random.shuffle(dict_key_ls)
    new_dic = {}
    new_dic['image'] = [_[0] for _ in dict_key_ls]
    new_dic['label'] = [_[1] for _ in dict_key_ls]
    return new_dic


def add_ddpm(x,y,dataset="cifar10", root="./data",rate=0.5):
    import time
    st = time.time()
    assert rate >=0.5
    npzfile = np.load(os.path.join(root,f"{dataset}_ddpm.npz"))
    # npzfile = random_dic(npzfile)
    add_x = npzfile['image']
    add_y = npzfile['label']
    # i = 0
    # total_x = []
    # total_y = []
    # x = [xi for xi in x]
    # y = [yi for yi in y]
    # add_x = [xi for xi in add_x]
    # add_y = [yi for yi in add_y]

    origin_len = len(x)
    total_num = int(origin_len / (1 - rate))
    x_total = np.zeros([total_num, *(x[0].shape)], dtype=np.int8)
    y_total = np.zeros([total_num,],dtype=np.int32)
    print('loading ddpm dataset')
    # inside_idxs = []
    origin_idxs = np.arange(0, origin_len) * 2
    other_idxs = np.arange(0, origin_len) * 2 + 1
    other_idxs2 = np.arange(origin_len*2+1, total_num)
    other_idxs = np.concatenate((other_idxs, other_idxs2), axis=None)
    x_total[origin_idxs] += np.array(x)
    x_total[other_idxs] += add_x[:len(other_idxs)]
    y_total[origin_idxs] = np.array(y)
    y_total[other_idxs] = add_y[:len(other_idxs)]
    assert len(other_idxs)+len(origin_idxs) == len(x_total)
    print(f'origin:{len(x)}, ddpm:{len(add_x)}')
    eps=time.time()-st
    # while True:
    #     if i % 2 == 0:
    #         total_x.append(x[0])
    #         total_y.append(y[0])
    #         x.pop(0)
    #         y.pop(0)
    #         # inside_idxs.append(len(total_x)-1)
    #     else:
    #         total_x.append(add_x[0])
    #         total_y.append(add_y[0])
    #         add_x.pop(0)
    #         add_y.pop(0)
    #     i += 1
    #     if len(x) == 0:
    #         total_x += add_x
    #         total_y += add_y
    #         break
    #     if len(add_x) == 0:
    #         total_x += x
    #         total_y += y
    #         break
    # x, y = np.array(total_x), np.array(total_y)

    # x, y = x[:total_num], y[:total_num]
    # print(y[:3])
    # print(type(y))
    # print(add_y[:3])
    # print(type(add_y))
    print(f'loaded used {eps} s!')
    return x_total.tolist(),y_total.tolist()

def get_dataset(dataset, root='./data', train=True, fitr=None,args=None, eval=False, close_trans=False):
    if 'imagenet' in dataset:
        return imagenet_utils.get_dataset(dataset, root, train, args=args)

    transform = get_transforms(dataset, train=train, is_tensor=False,close_trans=close_trans)
    lp_fitr   = None if fitr is None else get_filter(fitr)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))
    return data.Dataset(x, y, transform, lp_fitr)


def get_dataset_wo_eot(dataset, root='./data', train=True, fitr=None,args=None,eval=False):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_dataset(dataset, root, train)

    transform = get_transforms(dataset, train=train, is_tensor=False,wo_eot=True)
    lp_fitr   = None if fitr is None else get_filter(fitr)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'tiny-imagenet':
        target_set = data.datasetTinyImageNet(root=root, train=train, transform=transform)
        x, y = target_set.x, target_set.y
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    # if eval == False and args and args.load_ddpm and train == True:
    #     x, y = add_ddpm(x, y, dataset=dataset, root=args.ddpm_path, rate=args.ddpm_rate)
    return data.Dataset(x, y, transform, lp_fitr)

def get_indexed_loader(dataset, batch_size, root='./data', train=True,args=None):
    if "imagenet" in dataset:
        return imagenet_utils.get_indexed_loader(dataset, batch_size, root, train,args=args)

    target_set = get_dataset(dataset, root=root, train=train,close_trans=args.close_trans)

    if train:
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    else:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True,args=args)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False,args=args)

    return loader


def get_origin_x(dataset, batch_size,args=None, root='./data', train=True):
    # if dataset == 'imagenet' or dataset == 'imagenet-mini':
    #     return imagenet_utils.get_indexed_tensor_loader(dataset, batch_size, root, train)
    #
    target_set = get_dataset(dataset, root=root, train=train,eval=False, args=args)

    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)
    # here the ouput should still be within [0, 255]

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True,args=args)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False,args=args)

    return loader

def get_indexed_tensor_loader(dataset, batch_size,args=None, root='./data', train=True):
    if 'imagenet' in dataset:
        return imagenet_utils.get_indexed_tensor_loader(dataset, batch_size, root, train,args=args)

    target_set = get_dataset(dataset, root=root, train=train,eval=False, args=args)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)
    # here the ouput should still be within [0, 255]

    # if args.label_bias:
    #     # assert len(target_set.y[0]) == 1
    #     target_set.y = [(yi + np.random.randint(0,9)) % 10 for yi in target_set.y]

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True,args=args)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False,args=args)

    return loader


def get_shortcut_poisoned_loader(
        dataset, batch_size, root='./data', train=True, noise_rate=1.0,args=None,fitr=None):

    if 'imagenet' in dataset:
        # if args.shortcut:
        return imagenet_utils.get_poisoned_loader_shortcut(
                dataset, batch_size, root, train, noise_rate,fitr,args
            )
        # else: 
        #     return imagenet_utils.get_poisoned_loader(
        #         dataset, batch_size, root, train, noise_rate, fitr)

    target_set = get_dataset(dataset, root=root, train=train,eval=True,close_trans=args.close_trans, fitr=fitr)

    # if noise_path is not None:
    #     with open(noise_path, 'rb') as f:
    #         raw_noise = pickle.load(f)

    #     assert isinstance(raw_noise, np.ndarray)
    #     assert raw_noise.dtype == np.int8

    raw_noise = shortcut_noise(target_set.x, target_set.y, noise_frame_size=8, norm_ball=args.defense_budget).astype(np.int16)
    
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

    ''' add noise to images (uint8, 0~255) '''
    # target_set.origin_x = target_set.x
    imgs = target_set.x.astype(np.int16) + noise
    imgs = imgs.clip(0,255).astype(np.uint8)
    target_set.x = imgs
    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    # print('target y max min')
    # print(target_set.y.max(), target_set.y.min())

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True,args=args)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False,args=args)

    return loader



def get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0, poisoned_indices_path=None, fitr=None,close_trans=False,args=None):

    if 'imagenet' in dataset:
        return imagenet_utils.get_poisoned_loader(
                dataset, batch_size, root, train, noise_path, noise_rate, poisoned_indices_path, fitr,args)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr, eval=True,close_trans=args.close_trans)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)

        if poisoned_indices_path is not None:
            with open(poisoned_indices_path, 'rb') as f:
                indices = pickle.load(f)
        else:
            indices = np.random.permutation(len(noise))[:int(len(noise)*noise_rate)]

        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])
        
        if args.only_non_robust:
            target_set.x = target_set.x[np.random.permutation(len(target_set.x))]
        ''' add noise to images (uint8, 0~255) '''
        # target_set.origin_x = target_set.x
        imgs = target_set.x.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.x = imgs

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    # print('target y max min')
    # print(target_set.y.max(), target_set.y.min())

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True,args=args)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False,args=args)

    return loader


def get_clear_loader(
        dataset, batch_size, root='./data', train=True,
        noise_rate=1.0, poisoned_indices_path=None, fitr=None, args=None):

    if dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenet-two-class' or dataset == 'imagenet-ten-class':
        return imagenet_utils.get_clear_loader(
                dataset, batch_size, root, train, noise_rate, poisoned_indices_path, args=args)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr, args=args)
    data_nums = len(target_set)

    if poisoned_indices_path is not None:
        with open(poisoned_indices_path, 'rb') as f:
            poi_indices = pickle.load(f)
        indices = np.array( list( set(range(data_nums)) - set(poi_indices) ) )

    else:
        indices = np.random.permutation(range(data_nums))[: int( data_nums * (1-noise_rate) )]

    ''' select clear examples '''
    target_set.x = target_set.x[indices]
    target_set.y = np.array(target_set.y)[indices]

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    elif dataset == 'tiny-imagenet':
        in_dims, out_dims = 3, 200
    elif dataset == 'imagenet':
        in_dims, out_dims = 3, 1000
    elif dataset == 'imagenet-mini':
        in_dims, out_dims = 3, 100
    elif dataset == 'imagenet-two-class':
        in_dims, out_dims = 3, 2
    elif dataset == 'imagenet-ten-class':
        in_dims, out_dims = 3, 10
    elif dataset == 'imagenet-three-class':
        in_dims, out_dims = 3, 3
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    print('in_dims: {}, out_dims: {}'.format(in_dims, out_dims))

    if arch == 'resnet18':
        return models.resnet18(in_dims, out_dims)

    elif arch == 'resnet50':
        return models.resnet50(in_dims, out_dims)

    elif arch == 'wrn-34-10':
        return models.wrn34_10(in_dims, out_dims)

    elif arch == 'vgg11-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenet-two-class':
            raise NotImplementedError
        return models.vgg11_bn(in_dims, out_dims)

    elif arch == 'vgg16-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini' or dataset == 'imagenet-two-class':
            # raise NotImplementedError
            return models.img_vgg16_bn(in_dims, out_dims)
        return models.vgg16_bn(in_dims, out_dims)

    elif arch == 'vgg19-bn':
        return models.vgg19_bn(in_dims, out_dims)

    elif arch == 'densenet-121':
        return models.densenet121(num_classes=out_dims)

    else:
        raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/log.txt'.format(args.save_dir), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger


def evaluate(model, criterion, loader, cpu,args=None, aug=None,eval=False):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        # if not cpu: x, y = x.to(device), y.to(device)
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state


import torch
from torch.utils.data import Dataset
import numpy as np


import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path = path
        self.path = f"./tmp/{random.randint(0, 100)}-{random.randint(0, 10e9)}-ckpt.pth"
        self.trace_func = trace_func
        
        
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # load the last checkpoint with the best model
                model.load_state_dict(torch.load(self.path))
                os.remove(self.path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
