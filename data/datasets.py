"""
Classes to create datasets.

Author: Prisca Dotti
Last modified: 08.04.2024
"""

import logging
import math
import os
import random
from typing import Any, Dict, List, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import interp1d
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur

from config import TrainingConfig, config
from data.data_processing_tools import detect_spark_peaks, get_cell_mask, remove_padding
from utils.in_out_tools import load_annotations_ids, load_movies_ids

__all__ = [
    "CaEventsDataset",
    "CaEventsDatasetTempRed",
    "CaEventsDatasetResampled",
    "CaEventsDatasetLSTM",
    "CaEventsDatasetInference",
    "PatchCaEventsDataset",
]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


"""
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Class label filenames are: XX_class_label.tif
Event label filenames are: XX_event_label.tif
"""


class CaEventsDataset(Dataset):
    """
    A PyTorch Dataset class for ca release events detection.

    Args:
        params (TrainingConfig): A configuration object containing the
            dataset parameters.
        **kwargs: Additional keyword arguments to customize the dataset.

    Keyword Args:
        base_path (str): The base path to the dataset files on disk.
        sample_ids (List[str]): A list of sample IDs to load from disk.
        load_instances (bool): Whether to load instance data from disk.
        movies (List[np.ndarray]): A list of numpy arrays containing the movie
            data.
        labels (List[np.ndarray]): A list of numpy arrays containing the ground
            truth labels.
        instances (List[np.ndarray]): A list of numpy arrays containing the
            instance data.
        # stride (int): The stride to use when generating samples from the movie
        #     data.

    Raises:
        ValueError: If neither `movies` nor `base_path` and `sample_ids` are
        provided.

    Attributes:
        params (TrainingConfig): The configuration object containing the dataset
            parameters.
        window_duration (int): The size (t, y, or z) of each patch.
        window_stride (int): The stride to use when generating samples from the movie
            data.
        movies (List[torch.Tensor]): A list of PyTorch tensors containing the
            movie data.
        labels (List[torch.Tensor]): A list of PyTorch tensors containing the
            ground truth labels.
        instances (List[torch.Tensor]): A list of PyTorch tensors containing the
            instance data.
        gt_available (bool): Whether ground truth labels are available for the
            dataset.
        spark_peaks (List[Tuple[int, int]]): A list of tuples containing the
            (t, y, x) coordinates of the spark peaks in each movie.
        # original_shapes (List[int, int, int]): A list of the original shapes
        #     of the movies before padding.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx: int) -> Dict[str, Any]: Returns a dictionary containing
            the data, labels, and metadata for a given sample.
    """

    def __init__(self, params: TrainingConfig, **kwargs) -> None:
        # Get the dataset and training parameters
        self.params = params

        base_path: str = kwargs.get("base_path", "")
        sample_ids: List[str] = kwargs.get("sample_ids", [])

        movies: List[np.ndarray] = kwargs.get("movies", [])

        if base_path and sample_ids:
            # Load data from disk if base_path and sample_ids are provided
            self.base_path = base_path
            self.sample_ids = sample_ids
            load_instances: bool = kwargs.get("load_instances", False)

            ### Get video samples and ground truth ###
            movies = self._load_movies()  # dict of numpy arrays
            labels = self._load_labels()  # list of numpy arrays
            instances = (
                self._load_instances() if load_instances else []
            )  # list of numpy arrays

        elif movies:
            # Otherwise, data is provided directly
            labels: List[np.ndarray] = kwargs.get("labels", [])
            instances: List[np.ndarray] = kwargs.get("instances", [])

            # Create empty attributes
            self.base_path = ""
            self.sample_ids = []

        else:
            raise ValueError(
                "Either movies or base_path and sample_ids must be provided."
            )

        # Store the dataset parameters
        self.window_duration = params.data_duration
        self.window_stride: int = kwargs.get("stride", 0) or params.data_stride
        self.patch_duration = int(params.patch_size[0])
        self.patch_height = int(params.patch_size[1])
        self.patch_width = int(params.patch_size[2])

        # Store the movies, labels and instances
        self.movies = [torch.from_numpy(movie.astype(np.int32)) for movie in movies]
        self.labels = [torch.from_numpy(label.astype(np.int8)) for label in labels]
        self.instances = [
            torch.from_numpy(instance.astype(np.int16)) for instance in instances
        ]
        self.gt_available = True if len(labels) == len(movies) else False

        # Mask out regions outside the cell if required
        if params.mask_cell_exterior:
            self.labels = [
                self._mask_cell_exterior(movie, label)
                for movie, label in zip(self.movies, self.labels)
            ]

        # If instances are available, get the locations of the spark peaks
        if len(self.instances) > 0:
            self.spark_peaks = self._detect_spark_peaks()
        else:
            self.spark_peaks = []

        # Preprocess videos if necessary
        self._preprocess_videos()

        # # Store original duration of all movies before padding
        # self.original_shapes = [movie.shape for movie in self.movies]

        # # Adjust videos shape so that it is suitable for the model
        # self._adjust_videos_shape()

    ############################## Class methods ###############################

    def __len__(self) -> int:
        return len(self.movies)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.__len__() + idx

        sample_dict = {}

        # # Get the movie index and chunk index for the given idx
        # movie_idx, chunk_idx = self._get_movie_and_chunk_indices(idx)
        # sample_dict["movie_id"] = movie_idx

        # # Store the original duration of the movie
        # sample_dict["original_duration"] = self.original_shapes[movie_idx]

        # # Calculate the starting frame within the movie
        # start_frame = chunk_idx * self.stride
        # end_frame = start_frame + self.patch_size

        # Extract the windowed data and labels
        sample_dict["data"] = self.movies[idx]

        if self.gt_available:
            sample_dict["labels"] = self.labels[idx]

        # Add the sample ID (string) to the item dictionary, if available
        if self.sample_ids:
            sample_dict["sample_id"] = self.sample_ids[idx]

        return sample_dict

    def get_movies(self) -> Dict[int, np.ndarray]:
        """
        Returns the processed movies as a dictionary.

        Returns:
            dict: A dictionary containing the movies used as input to the model.
        """
        # Remove padding from the movies
        movies_numpy = {i: movie.numpy() for i, movie in enumerate(self.movies)}
        return movies_numpy

    def get_labels(self) -> Dict[int, np.ndarray]:
        """
        Returns the labels as a dictionary.

        Returns:
            dict: A dictionary containing the original labels used for training
            and testing.
        """
        # Remove padding from the labels
        labels_numpy = {i: label.numpy() for i, label in enumerate(self.labels)}
        return labels_numpy

    def get_instances(self) -> Dict[int, np.ndarray]:
        """
        Returns the instances as a dictionary.

        Returns:
            dict: A dictionary containing the original instances used for
            training and testing.
        """
        # Raise an error if instances are not available
        if not self.instances:
            raise ValueError("Instances not available for this dataset.")

        # Remove padding from the instances
        instances_numpy = {
            i: instance.numpy() for i, instance in enumerate(self.instances)
        }
        return instances_numpy

    # def set_debug_dataset(self) -> None:
    #     """
    #     Set the dataset to be a debug dataset by reducing the number of samples.
    #     """
    #     # For each movie, only keep the central chunk
    #     for i, movie in enumerate(self.movies):
    #         frames = movie.shape[0]
    #         samples_per_movie = (frames - self.patch_size) // self.stride + 1
    #         if samples_per_movie > 1:
    #             # Keep only the central chunk
    #             start_frame = (samples_per_movie // 2) * self.stride
    #             end_frame = start_frame + self.patch_size
    #             self.movies[i] = movie[start_frame:end_frame]
    #             if len(self.labels) > 0:
    #                 self.labels[i] = self.labels[i][start_frame:end_frame]
    #             if len(self.instances) > 0:
    #                 self.instances[i] = self.instances[i][start_frame:end_frame]
    #             self.original_shapes[i] = self.movies[i].shape[0]

    ############################## Private methods #############################

    def _load_movies(self) -> List[np.ndarray]:
        # Load movie data for each sample ID
        movies = load_movies_ids(
            data_folder=self.base_path,
            ids=self.sample_ids,
            names_available=True,
            movie_names="video",
        )

        # Extract and return the movie values as a list
        movies = list(movies.values())

        return movies

    def _load_labels(self) -> List[np.ndarray]:
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

        labels = load_annotations_ids(
            data_folder=self.base_path, ids=self.sample_ids, mask_names=mask_names
        )

        if labels:
            # Extract and return the mask values as a list
            labels = list(labels.values())
        else:
            labels = []

        return labels

    def _load_instances(self) -> List[np.ndarray]:
        # Load single event instances for each sample ID
        instances = load_annotations_ids(
            data_folder=self.base_path,
            ids=self.sample_ids,
            mask_names="event_label",
        )

        if instances:
            # Extract and return the mask values as a list
            instances = list(instances.values())
        else:
            raise ValueError("Instances not available for this dataset.")

        return instances

    # def _get_movie_and_chunk_indices(self, idx: int) -> Tuple[int, int]:
    #     """
    #     Given an index, returns the movie index and chunk index for the
    #     corresponding chunk in the dataset.

    #     Args:
    #         idx (int): The index of the chunk in the dataset.

    #     Returns:
    #         tuple: A tuple containing the movie index and chunk index for the
    #         corresponding chunk.
    #     """
    #     current_idx = 0  # Number of samples seen so far
    #     for movie_idx, movie in enumerate(self.movies):
    #         frames, _, _ = movie.shape
    #         samples_per_movie = (frames - self.patch_size) // self.stride + 1
    #         if idx < current_idx + samples_per_movie:
    #             # If idx is smaller than the number of samples seen so
    #             # far plus the number of samples in the current movie,
    #             # then the sample we're looking for is in the current
    #             # movie.
    #             chunk_idx = idx - current_idx  # chunk idx in the movie
    #             return movie_idx, chunk_idx

    #         current_idx += samples_per_movie

    #     # If the index is out of range, raise an error
    #     raise IndexError(
    #         f"Index {idx} is out of range for dataset of length {len(self)}"
    #     )

    def _detect_spark_peaks(self, class_name: str = "sparks") -> List[np.ndarray]:
        # Detect the spark peaks in the instance mask of each movie
        # Remark: can be used for other classes as well
        spark_peaks = []
        for movie, labels, instances in zip(self.movies, self.labels, self.instances):
            spark_mask = np.where(
                labels == config.classes_dict[class_name], instances, 0
            )
            self.coords_true = detect_spark_peaks(
                movie=movie.numpy(),
                instances_mask=spark_mask,
                sigma=config.sparks_sigma_dataset,
                max_filter_size=10,
            )
            spark_peaks.append(self.coords_true)
        return spark_peaks

    def _preprocess_videos(self) -> None:
        """
        Preprocesses the videos in the dataset.
        """
        if self.params.remove_background == "average":
            self.movies = [self._remove_background(movie) for movie in self.movies]

        if self.params.data_smoothing in ["2d", "3d"]:
            n_dims = int(self.params.data_smoothing[0])
            self.movies = [
                self._blur_movie(movie, n_dims=n_dims) for movie in self.movies
            ]

        if self.params.norm_video in ["movie", "abs_max", "std_dev"]:
            self.movies = [
                self._normalize(movie, norm_type=self.params.norm_video)
                for movie in self.movies
            ]

    def _remove_background(
        self, movie: torch.Tensor, mode: str = "average"
    ) -> torch.Tensor:
        if mode == "average":
            # Remove the average background from the video frames.
            background = torch.mean(movie, dim=0)
        elif mode == "moving":
            # Remove the moving average background from the video frames.
            T = movie.shape[0]
            N = self.window_duration // 2
            # Initialize an empty array for the background
            background = torch.zeros_like(movie)
            # Calculate the moving average for each frame
            for t in range(T):
                # Use slicing to calculate the moving average
                start = max(0, t - N)
                end = min(T, t + N)
                background[t] = torch.mean(movie[start:end], dim=0)
        else:
            raise ValueError(f"Unsupported background removal mode: {mode}")

        return movie - background

    def _blur_movie(self, movie: torch.Tensor, n_dims: int) -> torch.Tensor:
        # Define the kernel size and sigma based on the number of dimensions
        kernel_size = (3,) * n_dims
        sigma = 1.0

        # Apply gaussian blur to the video
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return gaussian_blur(movie)

    def _normalize(self, movie: torch.Tensor, norm_type: str) -> torch.Tensor:
        # Normalize the video frames.
        if norm_type == "movie":
            # Normalize each movie separately using its own max and min
            movie = (movie - torch.min(movie)) / (torch.max(movie) - torch.min(movie))
        elif norm_type == "abs_max":
            # Normalize each movie separately using the absolute max of uint16
            absolute_max = np.iinfo(np.uint16).max  # 65535
            movie = (movie - torch.min(movie)) / (absolute_max - torch.min(movie))
        elif norm_type == "std_dev":
            movie = movie.float()
            # Normalize each movie separately using its own standard deviation
            movie = (movie - torch.mean(movie)) / torch.std(movie)
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        return movie

    def _mask_cell_exterior(
        self, movie: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # Label regions outside the cell in the movie as ignored regions.
        outside = 1 - get_cell_mask(movie.numpy()).astype(int)
        outside = torch.from_numpy(outside) * config.ignore_index
        labels = torch.where(outside == 0, labels, outside)

        return labels


class CaEventsDatasetTempRed(CaEventsDataset):
    """
    A PyTorch Dataset class for spark detection with temporal reduction.

    This class is a subclass of the `SparkDataset` class and is specifically
    designed to work with deep learning models that use temporal reduction.
    It shrinks the annotation masks and instances to match the reduced temporal
    resolution of the model.

    Args:
        same as SparkDataset

    Raises:
        ValueError: If neither `movies` nor `base_path` and `sample_ids` are
        provided.
        AssertionError: If temporal reduction is not enabled in the parameters.
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # check that the temporal reduction is enabled in the parameters
        assert (
            params.temporal_reduction
        ), "Temporal reduction is not enabled in the parameters."

        # call the parent constructor
        super().__init__(params, **kwargs)

        # shrink the labels
        self.labels = [self._shrink_mask(mask) for mask in self.labels]

        # shrink the instances (not implemented yet!)
        if self.instances:
            # raise and error if instances are available
            raise NotImplementedError(
                "Instances are not supported for temporal reduction yet."
            )

    ############################## Private methods #############################

    def _shrink_mask(self, mask: np.ndarray) -> np.ndarray:
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
                        self._get_new_voxel_label(sub_mask[:, y, x])
                        for x in range(sub_mask.shape[2])
                    ]
                    for y in range(sub_mask.shape[1])
                ]
            )
            new_mask.append(new_frame)

        new_mask = np.stack(new_mask)
        return new_mask

    def _get_new_voxel_label(self, voxel_seq: np.ndarray) -> int:
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


class CaEventsDatasetResampled(CaEventsDataset):
    """
    Dataset class for resampled SR-calcium releases segmented dataset.

    This class extends the `SparkDataset` class and resamples the movies to a
    given frame rate. The original frame rate of the movies is obtained from
    their metadata. The resampled movies, labels, and instances are stored in
    memory.

    Args:
    - params (TrainingConfig): The training configuration.
    - movie_paths (List[str]): A list of paths to the movies (same order as the
        movies in the dataset). This allows to obtain the original frame rate of
        the movies from their metadata.
    - new_fps (int): The frame rate to resample the movies to.
    ... (same as SparkDataset)

    Raises:
        ValueError: If `movie_paths` or `new_fps` are not provided.
    """

    def __init__(
        self,
        params: TrainingConfig,
        **kwargs,
    ) -> None:
        # Verify that movie_paths and new_fps are provided
        if "movie_paths" not in kwargs:
            raise ValueError("movie_paths must be provided")
        if "new_fps" not in kwargs:
            raise ValueError("new_fps must be provided")

        self.movie_paths: List[str] = kwargs["movie_paths"]
        self.new_fps: int = kwargs["new_fps"]

        # Initialize SparkDataset class
        super().__init__(params=params, **kwargs)

    ############################## Class methods ###############################

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        # Get item from SparkDataset class and add the original frame rate
        item_dict = super().__getitem__(idx)
        item_dict["original_fps"] = self.original_fps[int(item_dict["movie_id"])]

        return item_dict

    ############################## Private methods #############################

    def _preprocess_videos(self) -> None:
        """
        Preprocesses the videos in the dataset.
        """
        # apply the same preprocessing as in SparkDataset class
        super()._preprocess_videos()

        # Get the original frame rate of the movies
        self.original_fps = [
            self._get_fps(movie_path) for movie_path in self.movie_paths
        ]

        # Resample the movies to the desired frame rate
        self.movies = [
            self._resample_video(movie, movie_path)
            for movie, movie_path in zip(self.movies, self.movie_paths)
        ]

        # Resample the labels to the desired frame rate
        if self.labels:
            self.labels = [
                self._resample_video(mask, movie_path)
                for mask, movie_path in zip(self.labels, self.movie_paths)
            ]

        # Resample the instances to the desired frame rate
        if self.instances:
            self.instances = [
                self._resample_video(instance, movie_path)
                for instance, movie_path in zip(self.instances, self.movie_paths)
            ]

    ####################### Methods for video resampling #######################

    def _resample_video(self, movie: torch.Tensor, movie_path: str) -> torch.Tensor:
        # Resample the video to the desired frame rate
        return self._video_spline_interpolation(
            movie=movie, movie_path=movie_path, new_fps=self.new_fps
        )

    def _get_fps(self, movie_path: str) -> float:
        """
        Compute estimated video FPS value with respect to sampling time deltas.

        Args:
            movie_path (str): Path to the video.

        Returns:
            float: Estimated FPS value.
        """
        times = self._get_times(movie_path)
        deltas = np.diff(times)
        return float(1.0 / np.mean(deltas))

    def _get_times(self, movie_path: str) -> np.ndarray:
        """
        Get times at which video frames were sampled.

        Args:
            movie_path (str): Path to the video.

        Returns:
            numpy.ndarray: Array of frame times.
        """
        with Image.open(movie_path) as img:
            exif_data = img.getexif()
        description = exif_data[270][0].split("\r\n")
        description = [line.split("\t") for line in description]
        description = [
            [int(i) if i.isdigit() else i for i in line] for line in description
        ]
        description = [d for d in description if isinstance(d[0], int)]
        return np.array([float(line[1]) for line in description])

    def _video_spline_interpolation(
        self, movie: torch.Tensor, movie_path: str, new_fps: int
    ) -> torch.Tensor:
        """
        Interpolate video frames based on new sampling times (FPS) using spline
        interpolation.

        Args:
            movie (numpy.ndarray): Input video frames.
            movie_path (str): Path to the video.
            new_fps (int): Desired FPS for the output video.

        Returns:
            numpy.ndarray: Interpolated video frames.
        """
        frames_time = self._get_times(movie_path)
        # Ensure movie is in numpy format for interpolation
        is_tensor = isinstance(movie, torch.Tensor)
        movie_np = movie.numpy() if is_tensor else movie

        # Use cubic spline interpolation
        f = interp1d(
            frames_time, movie_np, kind="cubic", axis=0, fill_value="extrapolate"
        )

        assert len(frames_time) == movie_np.shape[0], (
            "In video_spline_interpolation, the duration of the video "
            "is not equal to the number of frames."
        )

        frames_new = np.linspace(
            frames_time[0], frames_time[-1], int(frames_time[-1] * new_fps)
        )

        # Perform the interpolation and convert back to a torch.Tensor
        movie_interpolated = f(frames_new)
        if is_tensor:
            movie_interpolated = torch.from_numpy(movie_interpolated)

        return movie_interpolated


class CaEventsDatasetLSTM(CaEventsDataset):
    """
    SparkDataset class for UNet-convLSTM model.

    The dataset is adapted in such a way that each chunk is a sequence of
    frames centered around the frame to be predicted.
    The label is the segmentation mask of the central frame.
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # step = 1 and ignore_frames = 0 because we need to have a prediction
        # for each frame.
        self.params.data_stride = 1
        self.params.ignore_frames_loss = 0
        super().__init__(params, **kwargs)

    ############################## Class methods ###############################

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        As opposed to the SparkDataset class, here the label is just the
        middle frame of the chunk.
        """
        sample_dict = super().__getitem__(idx)

        if self.gt_available:
            # extract middle frame from label
            sample_dict["labels"] = sample_dict["labels"][self.window_duration // 2]

        return sample_dict

    ############################## Private methods #############################

    def _pad_short_video(
        self, video: torch.Tensor, padding_value: int = 0
    ) -> torch.Tensor:
        """
        Instead of padding the video with zeros, pad it with the first
        and last frame of the video.
        """
        padding_length = self.window_duration - video.shape[0]
        if padding_length:
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
                "replicate",
            )

            assert video.shape[0] == self.window_duration, "Padding is wrong"

            # logger.debug("Added padding to short video")

        return video

    def _pad_extremities_of_video(
        self, video: torch.Tensor, padding_value: int = 0
    ) -> torch.Tensor:
        """
        Pad duration/2 frames at the beginning and at the end of the video
        with the first and last frame of the video.
        """
        length = video.shape[0]

        # check that duration is odd
        assert self.window_duration % 2 == 1, "duration must be odd"

        pad = self.window_duration - 1

        # if video is int32, cast it to float32
        cast_to_float = video.dtype == torch.int32
        if cast_to_float:
            video = video.float()  # cast annotations to float32

        replicate = nn.ReplicationPad3d((0, 0, 0, 0, pad // 2, pad // 2))
        video = replicate(video[None, :])[0]

        if cast_to_float:
            video = video.int()  # cast annotations back to int32

        # check that duration of video is original duration + chunk duration - 1
        assert (
            video.shape[0] == length + self.window_duration - 1
        ), "padding at end of video is wrong"

        return video


class CaEventsDatasetInference(CaEventsDataset):
    """
    Create a dataset that contains only a single movie for inference.
    It requires either a single movie or a movie path to be provided.
    """

    def __init__(self, params: TrainingConfig, **kwargs) -> None:
        # Check that the arguments are suitable
        movie_path = kwargs.get("movie_path")
        movie = kwargs.get("movie")

        if movie is None and movie_path is None:
            raise ValueError("Either movie or movie_path must be provided.")
        if movie_path:
            # If a movie path is provided, load the movie from disk
            movies = [np.asarray(imageio.volread(movie_path))]
        else:
            movies = [movie]

        # Initialize SparkDataset class
        super().__init__(
            params=params, movies=movies, labels=[], instances=[], **kwargs
        )


### Dataset with sin channels ###
# re-implementation of some very old code (July 2020)


class CaEventsDatasetSinChannels(CaEventsDataset):
    """
    Create a dataset where each chunk is augmented with sinusoidal channels.

    Args:
    - params (TrainingConfig): The training configuration.
    - movie_paths (List[str]): A list of paths to the movies (same order as the
        movies in the dataset). This allows to obtain the original frame rate of
        the movies from their metadata.
    - n_sin_channels (List[int]): A list of 3 integers representing the number
        of sinusoidal channels to add along the time, y, and x dimensions.
    ... (same as SparkDataset)

    Raises:
        ValueError: If `n_sin_channels` is not provided.
    ... (same as SparkDataset)
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # Verify that n_sin_channels is provided
        if "n_sin_channels" not in kwargs:
            raise ValueError("n_sin_channels must be provided")

        self.n_sin_channels = kwargs["n_sin_channels"]

        # Check that n_sin_channels is a list of length 3
        assert len(self.n_sin_channels) == 3, "n_sin_channels must have length 3"

        # Initialize SparkDataset class
        super().__init__(params=params, **kwargs)

    ############################## Class methods ###############################

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        # Get item from the SparkDataset class and add the sinusoidal channels
        sample_dict = super().__getitem__(idx)
        sample_dict["data"] = self._concat_sin_channels(
            chunk=sample_dict["data"],
            t_freqs=self.n_sin_channels[0],
            y_freqs=self.n_sin_channels[1],
            x_freqs=self.n_sin_channels[2],
        )

        return sample_dict

    ############################## Private methods #############################

    def _concat_sin_channels(
        self, chunk: torch.Tensor, t_freqs: int, y_freqs: int, x_freqs: int
    ) -> torch.Tensor:
        """
        Concatenate sinusoidal channels with different frequencies to the input
        chunk.

        Args:
            chunk (torch.Tensor): Input chunk with shape frames x height (y) x
            width (x).

        Returns:
            torch.Tensor: Augmented chunk with sinusoidal patterns along
            dimension.
        """

        shape = chunk.shape

        t = torch.linspace(-np.pi, np.pi, shape[0])
        y = torch.linspace(-np.pi, np.pi, shape[1])
        x = torch.linspace(-np.pi, np.pi, shape[2])

        # Frequencies for sinusoidal patterns
        n_t = [2**i for i in range(0, t_freqs)]
        n_y = [2**i for i in range(0, y_freqs)]
        n_x = [2**i for i in range(0, x_freqs)]

        # Generate sinusoidal patterns
        t_sin = [torch.sin(n * t) for n in n_t]
        y_sin = [torch.sin(n * y) for n in n_y]
        x_sin = [torch.sin(n * x) for n in n_x]

        # Broadcast and transpose
        t_sin_all = [
            t_sin_n.expand(shape[1], shape[2], shape[0]).permute(2, 0, 1).unsqueeze(0)
            for t_sin_n in t_sin
        ]

        y_sin_all = [
            y_sin_n.expand(shape[2], shape[0], shape[1]).permute(1, 2, 0).unsqueeze(0)
            for y_sin_n in y_sin
        ]

        x_sin_all = [
            x_sin_n.expand(shape[0], shape[1], shape[2]).unsqueeze(0)
            for x_sin_n in x_sin
        ]

        return torch.cat(
            [chunk.unsqueeze(0)] + x_sin_all + y_sin_all + t_sin_all, dim=0
        )


### Datasets with patch extraction ###


class PatchCaEventsDataset(CaEventsDataset):
    """
    A PyTorch Dataset class for spark detection with patch extraction. The
    samples of this dataset are simply patches extracted from the movies. The
    information about the original frame rate of the movies is not stored, so
    it not suitable for inference.

    TODO:
    - improve sampling strategy, for the instance I am considering all valid
        points as centers
    - mask out regions outside the cell

    Args:
        same as SparkDataset
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # Initialize SparkDataset class
        super().__init__(params=params, **kwargs)

        # Set maximum number of patches for each training iteration
        self.max_patches = 10000

    ############################## Class methods ###############################

    def __len__(self) -> int:
        return self.max_patches

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        # Get a random center for the patch
        movie_idx: int = random.randint(0, len(self.movies) - 1)
        t_center: int = random.randint(
            self.patch_duration // 2,
            self.movies[movie_idx].shape[0] - self.patch_duration // 2,
        )
        y_center: int = random.randint(
            self.patch_height // 2,
            self.movies[movie_idx].shape[1] - self.patch_height // 2,
        )
        x_center: int = random.randint(
            self.patch_width // 2,
            self.movies[movie_idx].shape[2] - self.patch_width // 2,
        )

        # Extract the patch and label patch
        patch: torch.Tensor = self.movies[movie_idx][
            t_center - self.patch_duration // 2 : t_center + self.patch_duration // 2,
            y_center - self.patch_height // 2 : y_center + self.patch_height // 2,
            x_center - self.patch_width // 2 : x_center + self.patch_width // 2,
        ]

        if self.gt_available:
            label_patch: torch.Tensor = self.labels[movie_idx][
                t_center
                - self.patch_duration // 2 : t_center
                + self.patch_duration // 2,
                y_center - self.patch_height // 2 : y_center + self.patch_height // 2,
                x_center - self.patch_width // 2 : x_center + self.patch_width // 2,
            ]
        else:
            label_patch: torch.Tensor = torch.zeros([])

        # if the label only contains ignore_index, then the patch is invalid
        if torch.all(label_patch == config.ignore_index):
            return self.__getitem__(idx)

        return {
            "data": patch,
            "labels": label_patch,
            "movie_idx": movie_idx,
            "t_center": t_center,
            "y_center": y_center,
            "x_center": x_center,
        }


class PatchSparksDataset(CaEventsDataset):
    """
    A dataset class that partitions a video dataset into patches containing
    calcium sparks events. Each spark peak is contained into exactly one patch,
    ensuring that no sparks overlap between patches.

    Args:
        same as CaEventsDataset

    Additional attributes:
        patch_shape (tuple): The shape of each patch (t_patch, y_patch, x_patch).
        patches (list): A list of lists containing the start coordinates of
            patches for each sample.
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # Initialize SparkDataset class
        super().__init__(params=params, **kwargs)

        self.patch_shape = (self.patch_duration, self.patch_height, self.patch_width)
        self.patches = self._get_patches_from_movies()

        # Label puffs and waves in movie as undefined
        for event_type in ["puffs", "waves"]:
            self.labels = [
                torch.where(
                    label_mask == config.classes_dict[event_type],
                    config.ignore_index,
                    label_mask,
                )
                for label_mask in self.labels
            ]

    ############################## Class methods ###############################

    def __len__(self) -> int:
        return sum(len(patches) for patches in self.patches)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.__len__() + idx

        sample_dict = {}

        # Find the sample and patch index
        sample_idx, patch_idx = self._find_sample_and_patch_index(idx)

        # Get the corresponding patch start coordinates
        patch_start = self.patches[sample_idx][patch_idx]
        t_start, y_start, x_start = patch_start
        t_patch, y_patch, x_patch = self.patch_shape
        t_end = t_start + t_patch
        y_end = y_start + y_patch
        x_end = x_start + x_patch

        # Extract the windowed data and labels
        sample_dict["data"] = self.movies[sample_idx][
            t_start:t_end, y_start:y_end, x_start:x_end
        ]

        if self.gt_available:
            sample_dict["labels"] = self.labels[sample_idx][
                t_start:t_end, y_start:y_end, x_start:x_end
            ]

        if self.instances:
            sample_dict["instances"] = self.instances[sample_idx][
                t_start:t_end, y_start:y_end, x_start:x_end
            ]

        # Add the sample ID (string) to the item dictionary, if available
        if self.sample_ids:
            sample_dict["sample_id"] = self.sample_ids[sample_idx]

        sample_dict["patch_start"] = patch_start

        return sample_dict

    def get_movies(self) -> Dict[int, np.ndarray]:
        """
        Returns the processed movie patches as a dictionary.

        Returns:
            dict: A dictionary containing the patches used as input to the model.
        """
        patched_numpy = {
            i: self.__getitem__(i)["data"].numpy() for i in range(self.__len__())
        }
        return patched_numpy

    def get_labels(self) -> Dict[int, np.ndarray]:
        """
        Returns the labels as a dictionary.

        Returns:
            dict: A dictionary containing the original labels used for training
            and testing.
        """
        # Remove padding from the labels
        labels_numpy = {
            i: self.__getitem__(i)["labels"].numpy() for i in range(self.__len__())
        }
        return labels_numpy

    def get_instances(self) -> Dict[int, np.ndarray]:
        """
        Returns the instances as a dictionary.

        Returns:
            dict: A dictionary containing the original instances used for
            training and testing.
        """
        # Raise an error if instances are not available
        if not self.instances:
            raise ValueError("Instances not available for this dataset.")

        # Remove padding from the instances
        instances_numpy = {
            i: self.__getitem__(i)["instances"].numpy() for i in range(self.__len__())
        }
        return instances_numpy

    ############################## Private methods #############################

    def _find_sample_and_patch_index(self, idx: int) -> Tuple[int, int]:
        """
        Find the sample index and patch index within the sample for a given global index.

        Parameters:
        idx (int): The global index.

        Returns:
        tuple: (sample_index, patch_index) corresponding to the global index.
        """
        cumulative_patches = 0
        for sample_idx, patches in enumerate(self.patches):
            if cumulative_patches + len(patches) > idx:
                patch_idx = idx - cumulative_patches
                return sample_idx, patch_idx
            cumulative_patches += len(patches)
        raise IndexError(f"Index {idx} is out of range")

    def _fits_in_patch(self, spark, patch_start, patch_shape):
        """
        Check if a spark coordinate fits within the bounds of a given patch.

        Parameters:
        spark (tuple): The (t, y, x) coordinates of the spark.
        patch_start (tuple): The starting (t_start, y_start, x_start)
            coordinates of the patch.
        patch_shape (tuple): The shape (t_patch, y_patch, x_patch) of the patch.

        Returns:
        bool: True if the spark fits within the patch, False otherwise.
        """
        t_patch, y_patch, x_patch = patch_shape
        t, y, x = spark
        t_start, y_start, x_start = patch_start
        t_end = t_start + t_patch
        y_end = y_start + y_patch
        x_end = x_start + x_patch
        return t_start <= t < t_end and y_start <= y < y_end and x_start <= x < x_end

    def _find_patch_for_spark(self, spark, patches):
        """
        Find an existing patch that can contain the given spark.

        Parameters:
        spark (tuple): The (t, y, x) coordinates of the spark.
        patches (list): A list of patch start coordinates (t_start, y_start, x_start).

        Returns:
        tuple or None: The starting coordinates (t_start, y_start, x_start) of
        the patch if found, otherwise None.
        """
        for patch_start in patches:
            if self._fits_in_patch(spark, patch_start, self.patch_shape):
                return patch_start
        return None

    def _is_spark_assigned(self, spark):
        """
        Check if a spark has already been assigned to a patch.

        Parameters:
        spark (tuple): The (t, y, x) coordinates of the spark.

        Returns:
        bool: True if the spark is already assigned, False otherwise.
        """
        return spark in self._assigned_sparks

    def _create_patches_from_sparks(self, sparks_coord, video_shape, patch_shape):
        """
        Create patches from spark coordinates, ensuring each spark is contained in
        exactly one patch, and maximizing the number of patches.

        Parameters:
        sparks_coord (list): List of (t, y, x) coordinates of the sparks.
        video_shape (tuple): The shape (t, y, x) of the video.
        patch_shape (tuple): The shape (t_patch, y_patch, x_patch) of the patches.

        Returns:
        list: A list of patch start coordinates (t_start, y_start, x_start).
        """
        t_patch, y_patch, x_patch = patch_shape
        patches = []
        self._assigned_sparks = set()

        for spark in sparks_coord:
            if self._is_spark_assigned(spark):
                continue

            patch_start = self._find_patch_for_spark(spark, patches)
            if patch_start is None:
                t, y, x = spark
                t_start = max(0, t - t_patch // 2)
                y_start = max(0, y - y_patch // 2)
                x_start = max(0, x - x_patch // 2)

                if t_start + t_patch > video_shape[0]:
                    t_start = video_shape[0] - t_patch
                if y_start + y_patch > video_shape[1]:
                    y_start = video_shape[1] - y_patch
                if x_start + x_patch > video_shape[2]:
                    x_start = video_shape[2] - x_patch

                patch_start = (t_start, y_start, x_start)
                patches.append(patch_start)

            # Mark all sparks in the current patch as assigned
            t_start, y_start, x_start = patch_start
            t_end = t_start + t_patch
            y_end = y_start + y_patch
            x_end = x_start + x_patch

            for t, y, x in sparks_coord:
                if (
                    t_start <= t < t_end
                    and y_start <= y < y_end
                    and x_start <= x < x_end
                ):
                    self._assigned_sparks.add((t, y, x))

        return patches

    def _get_patches_from_movies(self) -> List[List[Tuple[int, int, int]]]:
        """
        Create patches from the spark coordinates in each movie.

        Returns:
        list: A list of lists of patch start coordinates (t_start, y_start, x_start).
        """
        patches_list = []
        for video, class_label, sparks_coord in zip(
            self.movies, self.labels, self.spark_peaks
        ):
            logger.debug(
                f"Getting patches from movie {len(patches_list) + 1}/{len(self.movies)}"
            )
            # Create patches from the spark coordinates
            patches = self._create_patches_from_sparks(
                sparks_coord=sparks_coord,
                video_shape=video.shape,
                patch_shape=self.patch_shape,
            )

            # Filter out patches that contain too much puff or wave pixels
            for patch_start in patches:
                t_start, y_start, x_start = patch_start
                t_end = t_start + self.patch_shape[0]
                y_end = y_start + self.patch_shape[1]
                x_end = x_start + self.patch_shape[2]
                puff_pixels = torch.sum(
                    class_label[t_start:t_end, y_start:y_end, x_start:x_end] == 2
                )
                wave_pixels = torch.sum(
                    class_label[t_start:t_end, y_start:y_end, x_start:x_end] == 3
                )
                if (
                    puff_pixels > 0.1 * self.patch_shape[1] * self.patch_shape[2]
                    or wave_pixels > 0.1 * self.patch_shape[1] * self.patch_shape[2]
                ):
                    logger.debug(
                        f"  Removing patch {patch_start} due to puff or wave pixels"
                    )
                    patches.remove(patch_start)

            patches_list.append(patches)

        return patches_list
