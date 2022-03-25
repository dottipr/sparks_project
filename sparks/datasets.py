'''
Classes to create training and testing datasets
'''

import os
import os.path
import glob
import logging

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
                           remove_avg_background, shrink_mask)


__all__ = ["SparkDataset", "SparkTestDataset"]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)


'''
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Annotation filenames are: XX_video_mask.tif
'''

class SparkDataset(Dataset):

    def __init__(self, base_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = 'average', temporal_reduction = False,
                 num_channels = 1, normalize_video = False,
                 only_sparks = False):

        # base_path is the folder containing the whole dataset (train and test)
        self.base_path = base_path

        # dataset parameters
        self.duration = duration
        self.step = step

        self.temporal_reduction = temporal_reduction
        if self.temporal_reduction:
            self.num_channels = num_channels

        self.normalize_video = normalize_video
        self.remove_background = remove_background

        # get videos and masks paths
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "videos", "[0-9][0-9]_video.tif")))
        self.annotations_files = sorted(glob.glob(os.path.join(self.base_path,
                                              "videos", "[0-9][0-9]_video_mask.tif")))

        # check that video filenames correspond to annotation filenames
        assert ((os.path.splitext(v) + "_mask") == os.path.splitext(a)
                for v,a in zip(self.files,self.annotations_files)), \
               "Video and annotation filenames do not match"

        # get videos and masks data
        self.data = [np.asarray(imageio.volread(file)) for file in self.files]
        self.annotations = [np.asarray(imageio.volread(f)).astype('int')
                            for f in self.annotations_files]

        # preprocess videos if necessary
        if self.remove_background == 'average':
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

        if self.normalize_video:
            self.data = [(video - video.min()) / (video.max() - video.min())
                         for video in self.data]

        # pad movies whose length does not match chunks_duration and step params
        self.data = [self.pad_end_of_video(video) for video in self.data]
        self.annotations = [self.pad_end_of_video(mask, mask=True) for mask in self.annotations]

        # pad movies shorter than chunk duration with zeros before beginning and after end
        self.data = [self.pad_short_video(video) for video in self.data]
        self.annotations = [self.pad_short_video(mask) for mask in self.annotations]

        # compute chunks indices
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()

        #print("movie shape", self.data[-1].shape)
        #print("movies' length", self.lengths)
        #print("num chunks", self.tot_blocks)
        #print("number of previous videos chunks", self.preceding_blocks)

        # if using temporal reduction, shorten the annotations duration
        if self.temporal_reduction:
            self.annotations = [shrink_mask(mask, self.num_channels)
                                for mask in self.annotations]

        #print("annotations shape", self.annotations[-1].shape)

        # if training with sparks only, set puffs and waves to 0
        if only_sparks:
            logger.info("Removing puff and wave annotations in training set")
            self.annotations = [np.where(np.logical_or(mask==1, mask==4),
                                         mask, 0) for mask in self.annotations]

    def pad_short_video(self, video):
        # pad videos shorter than chunk duration with zeros on both sides
        if video.shape[0] < self.duration:
            pad = self.duration - video.shape[0]
            video = np.pad(video, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))

            assert video.shape[0] == self.duration, "padding is wrong"

            logger.info("Added padding to short video")

        return video

    def pad_end_of_video(self, video, mask=False):
        # pad videos whose length does not match with chunks_duration and step params
        length = video.shape[0]
        if (((length-self.duration)/self.step) % 1 != 0):
            pad = (self.duration
                        + self.step*(1+(length-self.duration)//self.step)
                        - length)
            video = np.pad(video,((0,pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            length = video.shape[0]
            if not mask:
                logger.info(f"Added padding of {pad} frames to video with unsuitable duration")

        assert ((length-self.duration)/self.step) % 1 == 0, "padding at end of video is wrong"

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
        if idx < 0 : idx = self.__len__() + idx

        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]

        if self.remove_background == 'average':
            # remove the background of the single chunk
            # !! se migliora molto i risultati, farlo nel preprocessing che se
            #    no è super lento
            chunk = remove_avg_background(chunk)

        if not self.normalize_video:
            chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        assert chunk.min() >= 0 and chunk.max() <= 1, \
               "chunk values not normalized between 0 and 1"
        chunk = np.float32(chunk)
        #print("min and max value in chunk:", chunk.min(), chunk.max())

        #print("vid id", vid_id)
        #print("chunk id", chunk_id)
        #print("chunks", chunks[chunk_id])

        if self.temporal_reduction:
            assert self.lengths[vid_id] % self.num_channels == 0, \
                   "video length must be a multiple of num_channels"
            assert self.step % self.num_channels == 0, \
                   "step must be a multiple of num_channels"
            assert self.duration % self.num_channels == 0, \
                    "duration must be multiple of num_channels"

            masks_chunks = get_chunks(self.lengths[vid_id]//self.num_channels,
                                      self.step//self.num_channels,
                                      self.duration//self.num_channels)

            #print("mask chunk", masks_chunks[chunk_id])
            labels = self.annotations[vid_id][masks_chunks[chunk_id]]
        else:
            labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels

class SparkTestDataset(Dataset): # dataset that load a single video for testing

    def __init__(self, video_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = 'average', gt_available = True,
                 temporal_reduction = False, num_channels = 1,
                 normalize_video = False, only_sparks = False):

        # video_path is the complete path to the video
        # gt_available == True if ground truth annotations is available

        self.gt_available = gt_available

        self.temporal_reduction = temporal_reduction
        if self.temporal_reduction:
            self.num_channels = num_channels

        self.normalize_video = normalize_video
        self.remove_background = remove_background

        # get video path and array
        self.video_path = video_path
        self.video = imageio.volread(self.video_path)

        # get video name
        path, filename = os.path.split(video_path)
        self.video_name = os.path.splitext(filename)[0]

        # get mask path and array
        if self.gt_available:
            mask_filename = filename[:2]+"_video_mask.tif"
            mask_path = os.path.join(path, mask_filename)
            self.mask = imageio.volread(mask_path)

        # perform some preprocessing on videos, if required
        if self.remove_background == 'average':
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

        if self.normalize_video:
            self.video = (self.video-self.video.min())/(self.video.max()-self.video.min())

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
            self.length = self.video.shape[0]

            logger.info(f"Added padding of {self.pad} frames to video with unsuitable duration (test)")

        assert ((self.length-self.duration)/self.step) % 1 == 0, "padding at end of video is wrong (test)"

        # blocks in the video :
        self.blocks_number = ((self.length-self.duration)//self.step)+1

        # if using temporal reduction, shorten the annotations duration
        if self.temporal_reduction:
            self.mask = shrink_mask(self.mask, self.num_channels)

        # if training with sparks only, set puffs and waves to 0
        if only_sparks:
            logger.info("Removing puff and wave annotations in testing sample")
            self.mask = np.where(np.logical_or(self.mask==1, self.mask==4),
                                 self.mask, 0)


    def pad_short_video(self, video, mask):
        # pad videos shorter than chunk duration with zeros on both sides
        if self.length < self.duration:
            pad = self.duration - self.length
            video = np.pad(video, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))
            mask = np.pad(mask, ((pad//2, (pad%2) + (pad//2)), (0,0), (0,0)))

            self.length = video.shape[0]
            assert self.length == self.duration, "padding is wrong (test)"

            logger.info("Added padding to short video (test)")

        return video, mask

    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]

        if self.remove_background == 'average':
            # remove the background of the single chunk
            # !! se migliora molto i risultati, farlo nel preprocessing che se
            #    no è super lento
            chunk = remove_avg_background(chunk)

        if not self.normalize_video:
            chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        assert chunk.min() >= 0 and chunk.max() <= 1, \
               "chunk values not normalized between 0 and 1 (test)"
        chunk = np.float32(chunk)
        if self.gt_available:

            if self.temporal_reduction:
                assert self.length % self.num_channels == 0, \
                       "video length must be a multiple of num_channels (test)"
                assert self.step % self.num_channels == 0, \
                       "step must be a multiple of num_channels (test)"
                assert self.duration % self.num_channels == 0, \
                        "duration must be multiple of num_channels (test)"

                masks_chunks = get_chunks(self.length//self.num_channels,
                                          self.step//self.num_channels,
                                          self.duration//self.num_channels)

                labels = self.mask[masks_chunks[chunk_id]]
            else:
                labels = self.mask[chunks[chunk_id]]

            return chunk, labels

        return chunk
