# this code is based on the official version https://github.com/dayu11/Availability-Attacks-Create-Shortcuts/tree/d27ab29c5cc4f8da28200c98265dc2e2d45dfd79/synthetic_perturbations

import numpy as np
# import torch 
# import pandas as pd
from sklearn.datasets import make_classification


def comput_l2norm_lim(linf=0.03, feature_dim=3072):
    return np.sqrt(linf**2 * feature_dim)

def normalize_l2norm(data, norm_lim):
    n = data.shape[0]
    orig_shape = data.shape
    flatten_data = data.reshape([n, -1])
    norms = np.linalg.norm(flatten_data, axis=1, keepdims=True)
    flatten_data = flatten_data/norms
    data = flatten_data.reshape(orig_shape)
    data = data * norm_lim
    return data

def shortcut_noise(x, y, noise_frame_size, norm_ball, img_size=None, c=None):
    """
    input: image array, image label
    output: perturbed image and its label
    """
    num_classes = len(np.unique(y))
    y = np.array(y)
    x = np.array(x)
    n = x.shape[0]
    if img_size is None:
        img_size = x.shape[1]
    if c is None:
        c = x.shape[3]
    is_even = img_size % noise_frame_size  
    num_patch = img_size//noise_frame_size
    if(is_even > 0):
        num_patch += 1
    n_random_fea =  int((img_size/noise_frame_size)**2 * 3)
    
    # generate initial data points
    simple_data, simple_label = make_classification(n_samples=n, n_features=n_random_fea, n_classes=num_classes, n_informative=n_random_fea, n_redundant=0, n_repeated=0, class_sep=10., flip_y=0., n_clusters_per_class=1)
    simple_data = simple_data.reshape([simple_data.shape[0], num_patch, num_patch, 3])
    simple_data = simple_data.astype(np.float32)
    
    
    # duplicate each dimension to get 2-D patches
    simple_images = np.repeat(simple_data, noise_frame_size, 2) 
    simple_images = np.repeat(simple_images, noise_frame_size, 1)
    simple_data = simple_images[:, 0:img_size, 0:img_size, :]
    
    # project the synthetic images into a small L2 ball
    linf = norm_ball
    feature_dim = img_size**2 * 3
    l2norm_lim = comput_l2norm_lim(linf, feature_dim)
    simple_data = normalize_l2norm(simple_data, l2norm_lim)
    
    noise = np.zeros((n, img_size, img_size, c))
    
    # add synthetic noises to original examples
    for label in range(num_classes):
        orig_data_idx = y == label
        simple_data_idx = simple_label == label
        mini_simple_data = simple_data[simple_data_idx][0:int(sum(orig_data_idx))]
        noise[orig_data_idx] += mini_simple_data

    return noise