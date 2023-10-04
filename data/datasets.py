"""
Classes to create training and testing datasets

Author: Prisca Dotti
Last modified: 29.09.2023
"""

import logging
import math
import ntpath
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve
from torch import nn
from torch.utils.data import Dataset

from config import config
from data.data_processing_tools import detect_spark_peaks
from utils.in_out_tools import load_annotations_ids, load_movies_ids

__all__ = ["SparkDataset", "SparkDatasetPath", "SparkDatasetLSTM"]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)


"""
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Class label filenames are: XX_class_label.tif
Event label filenames are: XX_event_label.tif
"""


class SparkDataset(Dataset):
    def __init__(
        self,
        base_path,
        sample_ids,
        testing,
        params,
        resampling=False,
        resampling_rate=150,  # to be implemented
        gt_available=True,
        inference=None,
    ):
        """
        Dataset class for SR-calcium releases segmented dataset.

        Args:
            base_path (str): Directory where movies and annotation masks are saved.
            sample_ids (list): List of sample IDs used to create the dataset.
            testing (bool): If True, apply additional processing to data to compute
                metrics during validation/testing.
            params (TrainingConfig): TrainingConfig object containing training
                parameters.
            resampling (bool, optional): Not implemented yet. Default is False.
            resampling_rate (int, optional): Resampling rate used if resampling the
                movies. Default is 150.
            gt_available (bool, optional): True if sample's ground truth is available.
                Default is True.
            inference (str, optional): Inference mode used only during inference
                (not during training). Possible values are 'overlap' and 'average'.
                If 'overlap', overlapping frames are taken from preceding and
                following chunks.
                If 'average', the average is computed on overlapping chunks
                (TODO: define the best average method).
        """
        self.base_path = base_path
        self.params = params
        self.sample_ids = sample_ids
        self.testing = testing
        self.inference = inference
        self.gt_available = gt_available
        self.resampling = resampling
        self.resampling_rate = resampling_rate

        self.video_name = None
        self.pad = 0

        if self.inference:
            # Check that dataset contains a single video
            assert len(sample_ids) == 1, (
                f"Dataset set to inference mode, but it contains "
                f"{len(sample_ids)} samples: {sample_ids}."
            )

            # Check that inference mode is valid
            assert self.inference in ["overlap", "average"], (
                "If testing, select one inference mode from " "'overlap' and 'average'."
            )

            self.video_name = sample_ids[0]
            self.pad = 0

        ### Get video samples and ground truth ###
        self.data = [
            torch.from_numpy(movie.astype("int")) for movie in self.get_data()
        ]  # int32

        if self.gt_available:
            self.annotations = self.get_ground_truth()

        if self.inference:
            # Keep track of movie duration
            self.movie_duration = self.data[0].shape[0]

        if self.testing:
            assert self.gt_available, "If testing, ground truth must be available."
            assert len(sample_ids) == 1, (
                f"Dataset set to testing mode, but it contains "
                f"{len(sample_ids)} samples: {sample_ids}."
            )

        # Preprocess videos if necessary
        self.preprocess_videos()

        # Adjust videos shape so that it is suitable for the model
        self.adjust_videos_shape()

        # Compute chunks indices
        (
            self.lengths,
            self.tot_blocks,
            self.preceding_blocks,
        ) = self.compute_chunks_indices()

    ############################## class methods ###############################

    def get_data(self):
        # Load movie data for each sample ID
        movie_data = load_movies_ids(
            data_folder=self.base_path,
            ids=self.sample_ids,
            names_available=True,
            movie_names="video",
        )

        # Extract and return the movie values as a list
        data_list = list(movie_data.values())

        return data_list

    def preprocess_videos(self):
        if self.params.remove_background == "average":
            self.data = [self.remove_avg_background(video) for video in self.data]

        if self.params.data_smoothing == "2d":
            # apply gaussian blur to each frame of each video in self.data
            from torchvision.transforms import GaussianBlur

            gaussian_blur = GaussianBlur(kernel_size=(3, 3), sigma=1.0)
            self.data = [gaussian_blur(video) for video in self.data]

        elif self.params.data_smoothing == "3d":
            smooth_filter = 1 / 52 * torch.ones((3, 3, 3))
            smooth_filter[1, 1, 1] = 1 / 2
            self.data = [convolve(video, smooth_filter) for video in self.data]

        if self.resampling:
            self.fps = [self.get_fps(file) for file in self.files]
            self.data = [
                self.video_spline_interpolation(video, video_path, self.resampling_rate)
                for video, video_path in zip(self.data, self.files)
            ]

        if self.params.norm_video == "movie":
            self.data = [
                (video - video.min()) / (video.max() - video.min())
                for video in self.data
            ]

        elif self.params.norm_video == "abs_max":
            absolute_max = np.iinfo(np.uint16).max  # 65535
            self.data = [
                (video - video.min()) / (absolute_max - video.min())
                for video in self.data
            ]

    def adjust_videos_shape(self):
        self.data = [self.pad_short_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [
                self.pad_short_video(mask, padding_value=config.ignore_index)
                for mask in self.annotations
            ]

        self.data = [self.pad_extremities_of_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [
                self.pad_extremities_of_video(
                    mask, mask=True, padding_value=config.ignore_index
                )
                for mask in self.annotations
            ]

    def pad_short_video(self, video, padding_value=0):
        # Pad videos shorter than chunk duration with zeros on both sides
        padding_length = self.params.data_duration - video.shape[0]
        if padding_length > 0:
            video = F.pad(
                video,
                (
                    0,
                    0,
                    0,
                    0,
                    padding_length // 2,
                    padding_length // 2 + padding_length % 2,
                ),
                "constant",
                value=padding_value,
            )
            assert video.shape[0] == self.duration, "Padding is wrong"
            logger.debug("Added padding to short video")
        return video

    def pad_extremities_of_video(self, video, mask=False, padding_value=0):
        # Pad videos whose length does not match with chunks_duration and
        # step params
        length = video.shape[0]
        padding_length = self.params.data_step * math.ceil(
            (length - self.params.data_duration) / self.params.data_step
        ) - (length - self.params.data_duration)
        if padding_length > 0:
            # If testing, store the pad lenght as class attribute
            if self.inference is not None:
                self.pad = padding_length

            video = F.pad(
                video,
                (
                    0,
                    0,
                    0,
                    0,
                    padding_length // 2,
                    padding_length // 2 + padding_length % 2,
                ),
                "constant",
                value=padding_value,
            )
            length = video.shape[0]
            if not mask:
                logger.debug(
                    f"Added padding of {padding_length} frames to video with unsuitable duration"
                )

        assert (
            (length - self.duration) / self.params.data_step
        ) % 1 == 0, "padding at end of video is wrong"

        return video

    def get_ground_truth(self):
        # preprocess annotations if necessary
        if self.params.sparks_type == "raw":
            mask_names = "class_label"
        elif self.params.sparks_type == "peaks":
            mask_names = "class_label_peaks"
        elif self.params.sparks_type == "small":
            mask_names = "class_label_small_peaks"
        elif self.params.sparks_type == "dilated":
            mask_names = "class_label_dilated"
        else:
            raise NotImplementedError("Annotation type not supported yet.")

        annotations = list(
            load_annotations_ids(
                data_folder=self.base_path, ids=self.sample_ids, mask_names=mask_names
            ).values()
        )

        annotations = [torch.from_numpy(mask.astype("int8")) for mask in annotations]

        if self.testing:  # load event instances
            assert len(self.sample_ids) == 1, "Testing mode supports a single video."
            # if testing, the dataset contain a single video
            self.events = list(
                load_annotations_ids(
                    data_folder=self.base_path,
                    ids=self.sample_ids,
                    mask_names="event_label",
                ).values()
            )[0]
            self.events = torch.from_numpy(self.events.astype("int8"))

            logger.debug("Computing spark peaks...")
            spark_mask = np.where(annotations[0] == 1, self.events, 0)
            self.coords_true = detect_spark_peaks(
                movie=self.data[0],
                instances_mask=spark_mask,
                sigma=config.sparks_sigma_dataset,
                max_filter_size=10,
            )
            logger.debug(
                f"Sample {self.video_name} contains {len(self.coords_true)} sparks."
            )

        if self.temporal_reduction and self.gt_available:
            self.annotations = [self.shrink_mask(mask) for mask in self.annotations]

        if self.only_sparks and self.gt_available:
            logger.info("Removing puff and wave annotations in training set")
            annotations = [
                torch.where(torch.logical_or(mask == 1, mask == 4), mask, 0)
                for mask in annotations
            ]

        return annotations

    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # Blocks in each video :
        blocks_number = [
            ((length - self.params.data_duration) // self.params.data_step) + 1
            for length in lengths
        ]
        blocks_number = torch.as_tensor(blocks_number)
        # Number of blocks in preceding videos in data :
        preceding_blocks = torch.roll(torch.cumsum(blocks_number, dim=0), 1)
        tot_blocks = preceding_blocks[0].item()
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx

        # Index of video containing chunk idx
        vid_id = torch.where(self.preceding_blocks <= idx)[0][-1]
        # Index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]
        video_length = self.lengths[vid_id]

        chunks = self.get_chunks(
            video_length, self.params.data_step, self.params.data_duration
        )
        chunk = self.data[vid_id][chunks[chunk_id]]

        if self.params.remove_background == "moving":
            # Remove the background of the single chunk
            # !! If it significantly improves the results, do it during
            #    preprocessing, as otherwise, it's very slow.
            chunk = self.remove_avg_background(chunk)

        if self.params.norm_video == "chunk":
            chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())

        if self.gt_available:
            if self.params.temporal_reduction:
                masks_chunks = self.get_chunks(
                    video_length // self.params.num_channels,
                    self.params.data_step // self.params.num_channels,
                    self.params.data_duration // self.params.num_channels,
                )

                labels = self.annotations[vid_id][masks_chunks[chunk_id]]
            else:
                labels = self.annotations[vid_id][chunks[chunk_id]]

            return chunk, labels

        return chunk

    def remove_avg_background(self, video):
        # Remove the average background from the video frames.
        if torch.is_tensor(video):
            avg = torch.mean(video, axis=0)
        else:
            avg = np.mean(video, axis=0)
        return video - avg

    def get_chunks(self, video_length, step, duration):
        n_blocks = (video_length - duration) // step + 1

        # Returns a grid of indices for the chunks
        return torch.arange(duration)[None, :] + step * torch.arange(n_blocks)[:, None]

    ###################### Functions for video resampling ######################

    def get_times(self, video_path):
        """
        Get times at which video frames were sampled.

        Args:
            video_path (str): Path to the video.

        Returns:
            numpy.ndarray: Array of frame times.
        """
        description = Image.open(video_path).tag[270][0].split("\r\n")
        description = [line.split("\t") for line in description]
        description = [
            [int(i) if i.isdigit() else i for i in line] for line in description
        ]
        description = [d for d in description if isinstance(d[0], int)]
        return np.array([float(line[1]) for line in description])

    def get_fps(self, video_path):
        """
        Compute estimated video FPS value with respect to sampling time deltas.

        Args:
            video_path (str): Path to the video.

        Returns:
            float: Estimated FPS value.
        """
        times = self.get_times(video_path)
        deltas = np.diff(times)
        return 1 / np.mean(deltas)

    def video_spline_interpolation(self, video, video_path, new_fps=150):
        """
        Interpolate video frames based on new sampling times (FPS).

        Args:
            video (numpy.ndarray): Input video frames.
            video_path (str): Path to the video.
            new_fps (int): Desired FPS for the output video.

        Returns:
            numpy.ndarray: Interpolated video frames.
        """
        frames_time = self.get_times(video_path)
        f = interp1d(frames_time, video, kind="linear", axis=0)
        assert len(frames_time) == video.shape[0], (
            "In video_spline_interpolation the duration of the video "
            "is not equal to the number of frames"
        )
        frames_new = np.linspace(
            frames_time[0], frames_time[-1], int(frames_time[-1] * new_fps)
        )

        return f(frames_new)

    ##################### Functions for temporal reduction #####################

    def shrink_mask(self, mask):
        """
        Shrink an annotation mask based on the number of channels.

        Args:
            mask (numpy.ndarray): Input annotation mask.

        Returns:
            numpy.ndarray: Shrinked annotation mask.
        """
        assert (
            mask.shape[0] % self.params.num_channels == 0
        ), "Duration of the mask is not a multiple of num_channels."

        # Get tensor of duration 'self.num_channels'
        sub_masks = np.split(mask, mask.shape[0] // self.params.num_channels)
        new_mask = []

        # For each subtensor get a single frame
        for sub_mask in sub_masks:
            new_frame = np.array(
                [
                    [
                        self.get_new_voxel_label(sub_mask[:, y, x])
                        for x in range(sub_mask.shape[2])
                    ]
                    for y in range(sub_mask.shape[1])
                ]
            )
            new_mask.append(new_frame)

        new_mask = np.stack(new_mask)
        return new_mask

    def get_new_voxel_label(self, voxel_seq):
        """
        Get the new voxel label based on the sequence of voxel values.

        Args:
            voxel_seq (numpy.ndarray): Sequence of voxel values
                (num_channels elements).

        Returns:
            int: New voxel label.
        """
        # voxel_seq is a vector of 'num_channels' elements
        # {0} -> 0
        # {0, i}, {i} -> i, i = 1,2,3
        # {0, 1, i}, {1, i} -> 1, i = 2,3
        # {0, 2 ,3}, {2, 3} -> 3
        # print(voxel_seq)

        if np.max(voxel_seq == 0):
            return 0
        elif 1 in voxel_seq:
            return 1
        elif 3 in voxel_seq:
            return 3
        else:
            return np.max(voxel_seq)


# Define dataset same as SparkDataset, but load sample from a path


class SparkDatasetPath(SparkDataset):
    """
    SparkDataset class for UNet-convLSTM model.

    The dataset is adapted in such a way that each chunk is a sequence of
    frames centered around the frame to be predicted.
    The label is the segmentation mask of the central frame.
    """

    def __init__(self, sample_path, params, resampling=False, resampling_rate=150):
        self.sample_path = sample_path

        # get video name from path
        sample_name = ntpath.basename(sample_path).split(".")[0]

        # initialize parent class
        super().__init__(
            base_path=None,
            sample_ids=[sample_name],
            testing=False,
            params=params,
            resampling=resampling,
            resampling_rate=resampling_rate,
            gt_available=False,
            inference="overlap",  # can change this if necessary
        )

    def get_data(self):
        return [np.asarray(imageio.volread(self.sample_path))]


# define dataset class that will be used for training the UNet-convLSTM model


class SparkDatasetLSTM(SparkDataset):
    """
    SparkDataset class for UNet-convLSTM model.

    The dataset is adapted in such a way that each chunk is a sequence of
    frames centered around the frame to be predicted.
    The label is the segmentation mask of the central frame.
    """

    def __init__(
        self,
        base_path,
        sample_ids,
        testing,
        params,
        resampling=False,
        resampling_rate=150,
        gt_available=True,
        inference=None,
    ):
        """
        step = 1 and ignore_frames = 0 because we need to have a prediction
        for each frame.
        """

        super().__init__(
            base_path=base_path,
            sample_ids=sample_ids,
            testing=testing,
            params=params,
            resampling=resampling,
            resampling_rate=resampling_rate,
            gt_available=gt_available,
            inference=inference,
        )

    def pad_short_video(self, video, padding_value=0):
        """
        Instead of padding the video with zeros, pad it with the first
        and last frame of the video.
        """

        if video.shape[0] < self.params.data_duration:
            pad = self.params.data_duration - video.shape[0]
            video = F.pad(
                video, (0, 0, 0, 0, pad // 2, pad // 2 + pad % 2), "replicate"
            )

            assert video.shape[0] == self.params.data_duration, "padding is wrong"

            logger.debug("Added padding to short video")

        return video

    def pad_extremities_of_video(self, video, mask=False, padding_value=0):
        """
        Pad duration/2 frames at the beginning and at the end of the video
        with the first and last frame of the video.
        """
        length = video.shape[0]

        # check that duration is odd
        assert self.params.data_duration % 2 == 1, "duration must be odd"

        pad = self.params.data_duration - 1

        # if testing, store the pad lenght as class attribute
        if self.testing:
            self.pad = pad

        if mask:
            video = video.float()  # cast annotations to float32

        replicate = nn.ReplicationPad3d((0, 0, 0, 0, pad // 2, pad // 2))
        video = replicate(video[None, :])[0]

        if mask:
            video = video.int()  # cast annotations back to int32

        # check that duration of video is original duration + chunk duration - 1
        assert (
            video.shape[0] == length + self.params.data_duration - 1
        ), "padding at end of video is wrong"

        return video

    def __getitem__(self, idx):
        """
        As opposed to the SparkDataset class, here the label is just the
        middle frame of the chunk.
        """
        sample = super().__getitem__(idx)

        if self.gt_available:
            # extract middle frame from label
            label = sample[1][self.params.data_duration // 2]
            sample = (sample[0], label)

        return sample
