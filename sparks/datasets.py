'''
Classes to create training and testing datasets
'''

import os
import os.path
import glob

import imageio
import pandas as pd
import ntpath

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.ndimage.filters import convolve

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from dataset_tools import (get_chunks, get_fps, video_spline_interpolation,
                           remove_avg_background)


__all__ = ["SparkDataset", "SparkTestDataset"]


basepath = os.path.dirname("__file__")


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


'''
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Annotation filenames are: XX_video_mask.tif
'''

class SparkDataset(Dataset):

    def __init__(self, base_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False):

        # base_path is the folder containing the whole dataset (train and test)
        self.base_path = base_path

        # get videos and masks paths
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "videos", "*[!_mask].tif")))
        self.annotations_files = sorted(glob.glob(os.path.join(self.base_path,
                                              "videos", "*_mask.tif")))

        # check that video filenames correspond to annotation filenames
        assert ((os.path.splitext(v) + "_mask") == os.path.splitext(a)
                for v,a in zip(self.files,self.annotations_files)), \
               "Video and annotation filenames do not match"

        # get videos and masks data
        self.data = [np.asarray(imageio.volread(file)) for file in self.files]
        self.annotations = [np.asarray(imageio.volread(f)).astype('int')
                            for f in self.annotations_files]

        # preprocess videos if necessary
        if remove_background:
            self.data = [remove_avg_background(video) for video in self.data]
        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.data = [np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in video])
                                            for video in self.data]
        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(video, _smooth_filter) for video in self.data]
        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]

        # pad movies shorter than chunk duration with zeros before beginning and after end
        self.duration = duration
        self.data = [self.pad_short_video(video) for video in self.data]
        self.annotations = [self.pad_short_video(mask) for mask in self.annotations]

        # compute chunks indices
        self.step = step
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()

    def pad_short_video(self, video):
        # pad videos shorter than chunk duration with zeros on both sides
        if video.shape[0] < self.duration:
            pad = self.duration - video.shape[0]
            video = np.pad(video, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))

            assert video.shape[0] == self.duration, "padding is wrong"

        return video

    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # blocks in each video :
        blocks_number = [((length-self.duration)//self.step)+1
                         for length in lengths]
        # number of blocks in preceding videos in data :
        preceding_blocks = np.roll(np.cumsum(blocks_number),1)
        tot_blocks = preceding_blocks[0]
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels

class SparkTestDataset(Dataset): # dataset that load a single video for testing

    def __init__(self, video_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False, gt_available = True):

        # video_path is the complete path to the video
        # gt_available == True if ground truth annotations is available

        self.gt_available = gt_available

        # get video path and array
        self.video_path = video_path
        self.video = imageio.volread(self.video_path)

        # get mask path and array
        if self.gt_available:
            filename, ext = os.path.splitext(video_path)
            self.mask_path = filename + "_mask" + ext
            self.mask = imageio.volread(self.mask_path)

        self.video_name = path_leaf(filename)

        # perform some preprocessing on videos, if required
        if remove_background:
            self.video = remove_avg_background(self.video)
        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.video = np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in self.video])
        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.video = convolve(self.video, _smooth_filter)
        if resampling:
            self.fps = get_fps(self.file)
            self.video = video_spline_interpolation(self.video, self.file,
                                                    resampling_rate)

        self.step = step
        self.duration = duration
        self.length = self.video.shape[0]

        # pad movies shorter than chunk duration with zeros before beginning and after end
        self.video, self.mask = self.pad_short_video(self.video, self.mask)

        # if necessary, pad empty frames at the end
        self.pad = 0
        if (((self.length-self.duration)/self.step) % 1 != 0):
            self.pad = (self.duration
                        + self.step*(1+(self.length-self.duration)//self.step)
                        - self.length)
            self.video = np.pad(self.video,((0,self.pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            self.mask = np.pad(self.mask,((0,self.pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            self.length = self.length + self.pad

        # blocks in the video :
        self.blocks_number = ((self.length-self.duration)//self.step)+1

    def pad_short_video(self, video, mask):
        # pad videos shorter than chunk duration with zeros on both sides
        if self.length < self.duration:
            pad = self.duration - self.length
            video = np.pad(video, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))
            mask = np.pad(mask, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))

            self.length = video.shape[0]
            assert self.length == self.duration, "padding is wrong"

        return video, mask

    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)
        if self.gt_available:
            labels = self.mask[chunks[chunk_id]]

            return chunk, labels

        return chunk
