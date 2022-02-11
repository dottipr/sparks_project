"""
Functions needed to preprocess the csv files, create the structure of data in
the spark datasets and compute the weights of the network.
"""


import numpy as np
import torch
from torch import nn
from scipy import ndimage
from scipy.interpolate import interp1d
from PIL import Image

import torch


__all__ = ["get_chunks",
           "random_flip",
           "compute_class_weights",
           "weights_init",
           "get_times",
           "get_fps",
           "video_spline_interpolation",
           "remove_avg_background",
           "shrink_mask",
           "get_new_voxel_label"]


### functions for data preproccesing ###
def get_chunks(video_length, step, duration):
    n_blocks = ((video_length-duration)//(step))+1

    return np.arange(duration)[None,:] + step*np.arange(n_blocks)[:,None]


def random_flip(x, y):

    if np.random.uniform() > 0.5:
        x = x[..., ::-1]
        y = y[..., ::-1]

    if np.random.uniform() > 0.5:
        x = x[..., ::-1, :]
        y = y[..., ::-1, :]

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    return x, y


def remove_avg_background(video):
    # remove average background
    avg = np.mean(video, axis = 0)
    return np.add(video, -avg)


### functions related to U-Net hyperparameters ###


def compute_class_weights(dataset, w0=1, w1=1, w2=1, w3=1):
    # For 4 classes
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with torch.no_grad():
        for _,y in dataset:
            count0 += np.count_nonzero(y==0)
            count1 += np.count_nonzero(y==1)
            count2 += np.count_nonzero(y==2)
            count3 += np.count_nonzero(y==3)

    total = count0 + count1 + count2 + count3

    w0_new = w0*total/(4*count0) if count0 != 0 else 0
    w1_new = w1*total/(4*count1) if count1 != 0 else 0
    w2_new = w2*total/(4*count2) if count2 != 0 else 0
    w3_new = w3*total/(4*count3) if count3 != 0 else 0

    weights = np.array([w0_new, w1_new, w2_new, w3_new])

    return np.float64(weights)


def weights_init(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        stdv = np.sqrt(2/m.weight.size(1))
        m.weight.data.normal_(m.weight, std=stdv)


### functions for video resampling ###


def get_times(video_path):
    # get times at which video frames where sampled
    description = Image.open(video_path).tag[270][0].split('\r\n')
    description  = [line.split('\t') for line in description]
    description = [[int(i) if i.isdigit() else i for i in line] for line in description]
    description = [d for d in  description if isinstance(d[0], int)]
    return np.array([float(line[1]) for line in description])


def get_fps(video_path):
    # compute estimated video fps value wrt sampling times deltas
    times = get_times(video_path)
    deltas = np.diff(times)
    return 1/np.mean(deltas)


def video_spline_interpolation(video, video_path, new_fps=150):
    # interpolate video wrt new sampling times
    frames_time = get_times(video_path)
    f = interp1d(frames_time, video, kind='linear', axis=0)
    assert(len(frames_time) == video.shape[0])
    frames_new = np.linspace(frames_time[0], frames_time[-1], int(frames_time[-1]*new_fps))
    return f(frames_new)

def shrink_mask(mask, num_channels):
    # input is an annotation mask with the number of channels of the unet
    # output is a shrinked mask where :
    # {0} -> 0
    # {0, i}, {i} -> i, i = 1,2,3
    # {0, 1, i}, {1, i} -> 1, i = 2,3
    # {0, 2 ,3}, {2, 3} -> 2
    # and each voxel in the output corresponds to 'num_channels' voxels in
    # the input

    assert mask.shape[0] % num_channels == 0, \
    "in shrink_mask the duration of the mask is not a multiple of num_channels"

    # get subtensor of duration 'num_channels'
    sub_masks = np.split(mask, mask.shape[0]//num_channels)

    #print(sub_masks[0].shape)
    #print(len(sub_masks))

    new_mask = []
    # for each subtensor get a single frame
    for sub_mask in sub_masks:
        new_frame = np.array([[get_new_voxel_label(sub_mask[:,y,x]) for x in range(sub_mask.shape[2])] for y in range(sub_mask.shape[1])])
        #print(new_frame.shape)
        new_mask.append(new_frame)

    new_mask = np.stack(new_mask)
    return new_mask

def get_new_voxel_label(voxel_seq):
    # voxel_seq is a vector of 'num_channels' elements
    # {0} -> 0
    # {0, i}, {i} -> i, i = 1,2,3
    # {0, 1, i}, {1, i} -> 1, i = 2,3
    # {0, 2 ,3}, {2, 3} -> 3
    #print(voxel_seq)

    if np.max(voxel_seq == 0):
        return 0
    elif 1 in voxel_seq:
        return 1
    elif 3 in voxel_seq:
        return 3
    else:
        return np.max(voxel_seq)
