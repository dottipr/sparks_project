'''
Classes to create training and testing datasets
'''

import os
import os.path
import glob
import logging
import time

import imageio
import pandas as pd
import ntpath

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.ndimage.filters import convolve

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from dataset_tools import (get_chunks, get_fps, video_spline_interpolation,
                           remove_avg_background, shrink_mask, get_new_mask,
                           detect_spark_peaks,
                           load_movies_ids, load_annotations_ids
                           )
from metrics_tools import get_sparks_locations_from_mask


__all__ = ["SparkDataset"]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)


'''
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Class label filenames are: XX_class_label.tif
Event label filenames are: XX_event_label.tif
'''

class SparkDataset(Dataset):

    def __init__(self, base_path, sample_ids, testing,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = 'average', temporal_reduction = False,
                 num_channels = 1, normalize_video = 'chunk',
                 only_sparks = False, sparks_type = 'peaks', ignore_index=4
                 ignore_frames = 0, gt_available = True):

        '''
        Dataset class for SR-calcium releases segmented dataset.

        base_path:          directory where movies and annotation masks are
                            saved
        sample_ids:         list of sample IDs used to create the dataset
        testing:            if True, apply additional processing to data to
                            compute metrics during validation/testing
        step:               step between two chunks extracted from the sample
        duration:           duration of a chunk
        smoothing:          if '2d' or '3d', preprocess movie with simple
                            convolution (probably useless)
        resampling_rate:    resampling rate used if resampling the movies
        remove_background:  if 'moving' or 'average', remove background from
                            input movies accordingly
        temporal_reduction: set to True if using TempRedUNet (sample processed
                            in conv layers before unet)
        num_channels:       >0 if using temporal_reduction, value depends on
                            temporal reduction configuration
        normalize_video:    if 'chunk', 'movie' or 'abs_max' normalize input
                            video accordingly
        only_sparks:        if True, train using only sparks annotations
        sparks_type:        can be 'raw' or 'peaks' (use smaller annotated ROIs)
        ignore_frames:      if testing, used to ignore events in first and last
                            frames
        gt_available:       True if sample's ground truth is available
        '''

        # base_path is the folder containing the whole dataset (train and test)
        self.base_path = base_path

        # physiological params (for spark peaks results)
        self.pixel_size = 0.2 # 1 pixel = 0.2 um x 0.2 um
        self.min_dist_xy = round(1.8 / self.pixel_size) # min distance in space
        self.time_frame = 6.8 # 1 frame = 6.8 ms
        self.min_dist_t = round(20 / self.time_frame) # min distance in time

        # dataset parameters
        self.testing = testing
        self.gt_available = gt_available
        self.sample_ids = sample_ids
        self.only_sparks = only_sparks
        self.sparks_type = sparks_type
        self.ignore_index = ignore_index

        self.duration = duration
        self.step = step

        # if testing, get video name and take note of the padding applied
        # to the movie
        if self.testing:
            # if testing, the dataset contains a single video
            assert len(sample_ids)==1, "Dataset set to testing mode, but it contains more than one sample."

            # if testing, ground truth must be available
            assert gt_available, "If testing, ground truth must be available."

            self.video_name = sample_ids[0]
            self.pad = 0

        self.temporal_reduction = temporal_reduction
        if self.temporal_reduction:
            self.num_channels = num_channels

        self.normalize_video = normalize_video
        self.remove_background = remove_background

        # get video samples
        self.data = list(load_movies_ids(data_folder=self.base_path,
                                    ids=sample_ids,
                                    names_available=True,
                                    movie_names="video"
                                    ).values())
        self.data = [torch.from_numpy(movie.astype('int'))
                     for movie in self.data] # int32

        # get annotation masks, if ground truth is available:
        if self.gt_available:
            # get class label masks
            self.annotations = list(load_annotations_ids(
                                        data_folder=self.base_path,
                                        ids=sample_ids,
                                        mask_names="class_label"
                                        ).values())
            self.annotations = [torch.from_numpy(mask)
                                for mask in self.annotations] # int8

            # preprocess annotations if necessary
            assert self.sparks_type in ['peaks', 'smaller', 'raw'], "Sparks type should be 'peaks', 'smaller' or 'raw'."

            if self.sparks_type == 'peaks':
                # TODO: if necessary implement mask that contain only spark peaks
                pass
            elif self.sparks_type == 'smaller':
                # reduce the size of sparks annotations and replace difference with
                # an undefined label (4)

                # TODO: ...........
                # self.annotations = [process(mask) for mask is self.annotations]
                pass

            # if testing, load the event label masks too (for peaks detection)
            # and compute the location of the spark peaks
            if self.testing:
                # if testing, the dataset contain a single video

                # if testing, need to keep track of movie duration, in case it
                # is shorter than `chunks_duration and a pad is added
                self.movie_duration = (self.data[0]).shape[0]

                self.events = list(load_annotations_ids(
                                        data_folder=self.base_path,
                                        ids=sample_ids,
                                        mask_names="event_label"
                                        ).values())
                self.events = [torch.from_numpy(mask)
                                    for mask in self.events] # int8

                logger.info("Computing spark peaks...")
                self.coords_true = detect_spark_peaks(movie=self.data[0],
                                                      event_mask=self.events[0],
                                                      class_mask=self.annotations[0],
                                                      sigma=2,
                                                      max_filter_size=10)
                logger.info(f"Sample {self.video_name} contains {len(self.coords_true)} sparks.")

        # preprocess videos if necessary
        if self.remove_background == 'average':
            self.data = [remove_avg_background(video) for video in self.data]

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.data = [torch.from_numpy([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in video])
                                            for video in self.data]
        if smoothing == '3d':
            _smooth_filter = 1/52*torch.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(video, _smooth_filter) for video in self.data]

        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]

        if self.normalize_video == 'movie':
            self.data = [(video - video.min()) / (video.max() - video.min())
                         for video in self.data]
        elif self.normalize_video == 'abs_max':
            absolute_max = np.iinfo(np.uint16).max # 65535
            self.data = [(video-video.min())/(absolute_max-video.min())
                         for video in self.data]



        # pad movies shorter than chunk duration with zeros before beginning and after end
        self.data = [self.pad_short_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [self.pad_short_video(mask, padding_value=ignore_index)
                                for mask in self.annotations]

        # pad movies whose length does not match chunks_duration and step params
        self.data = [self.pad_end_of_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [self.pad_end_of_video(mask, mask=True, padding_value=ignore_index)
                                for mask in self.annotations]

        # compute chunks indices
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()


        # if using temporal reduction, shorten the annotations duration
        if self.temporal_reduction and self.gt_available:
            self.annotations = [shrink_mask(mask, self.num_channels)
                                for mask in self.annotations]

        #print("annotations shape", self.annotations[-1].shape)

        # if training with sparks only, set puffs and waves to 0
        if self.only_sparks and self.gt_available:
            logger.info("Removing puff and wave annotations in training set")
            self.annotations = [torch.where(torch.logical_or(mask==1, mask==4),
                                         mask, 0) for mask in self.annotations]

    def pad_short_video(self, video, padding_value=0):
        # pad videos shorter than chunk duration with zeros on both sides
        if video.shape[0] < self.duration:
            pad = self.duration - video.shape[0]
            video = F.pad(video,(0,0,0,0,pad//2,pad//2+pad%2),
                                'constant',value=padding_value)

            assert video.shape[0] == self.duration, "padding is wrong"

            logger.info("Added padding to short video")

        return video

    def pad_end_of_video(self, video, mask=False, padding_value=0):
        # pad videos whose length does not match with chunks_duration and
        # step params
        length = video.shape[0]
        if (((length-self.duration)/self.step) % 1 != 0):
            pad = (self.duration
                        + self.step*(1+(length-self.duration)//self.step)
                        - length)

            # if testing, store the pad lenght as class attribute
            if self.testing:
                self.pad = pad

            video = F.pad(video,(0,0,0,0,pad//2,pad//2+pad%2),
                                'constant',value=padding_value)
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
        blocks_number = torch.tensor(blocks_number)
        # number of blocks in preceding videos in data :
        preceding_blocks = torch.roll(torch.cumsum(blocks_number, dim=0),1)
        tot_blocks = preceding_blocks[0].detach().item()
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        if idx < 0 : idx = self.__len__() + idx

        #index of video containing chunk idx
        vid_id = torch.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]

        if self.remove_background == 'moving':
            # remove the background of the single chunk
            # !! se migliora molto i risultati, farlo nel preprocessing che se
            #    no è super lento
            chunk = remove_avg_background(chunk)

        if self.normalize_video == 'chunk':
            chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        assert chunk.min() >= 0 and chunk.max() <= 1, \
               "chunk values not normalized between 0 and 1"
        #print("min and max value in chunk:", chunk.min(), chunk.max())

        #print("vid id", vid_id)
        #print("chunk id", chunk_id)
        #print("chunks", chunks[chunk_id])

        if self.gt_available:
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

        return chunk
