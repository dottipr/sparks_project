"""
Functions that are used during the training of the neural network model.

Author: Prisca Dotti
Last modified: 17.10.2023
"""

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
from scipy import ndimage as ndi
from torch import nn

from config import TrainingConfig, config
from data.data_processing_tools import process_raw_predictions, remove_padding
from data.datasets import SparkDataset, SparkDatasetInference
from models.UNet import unet
from models.UNet.unet.trainer import _write_results

__all__ = [
    "MyTrainingManager",
    "TransformedSparkDataset",
    "training_step",
    "sampler",
    "random_flip",
    "random_flip_noise",
    "compute_class_weights",
    "compute_class_weights_instances",
    "weights_init",
    "run_dataloader_in_network",
    "do_inference",
    "max_inference",
    "gaussian_inference",
    "average_inference",
    "from_chunks_to_frames",
    "get_half_overlap",
    "overlap_inference",
    "get_raw_preds_dict",
    "test_function",
    "get_preds_from_path",
]

logger = logging.getLogger(__name__)


################################ Training step #################################


# Make one step of the training (update parameters and compute loss)
def training_step(
    dataset_loader: torch.utils.data.DataLoader,
    params: TrainingConfig,
    sampler: Callable[[torch.utils.data.DataLoader], dict],
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    # scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, torch.Tensor]:
    """
    Perform one training step.

    Args:
        dataset_loader (DataLoader): DataLoader for the training dataset.
        params (TrainingConfig): A TrainingConfig containing various parameters.
        sampler (callable): Function to sample data from the dataset.
        network (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

    Returns:
        dict: A dictionary containing the training loss.
    """
    network.train()

    # Sample data from the dataset
    sample = sampler(dataset_loader)
    x = sample["data"]
    y = sample["labels"]

    x = x.to(params.device, non_blocking=True)  # [b, d, 64, 512]
    # [b, d, 64, 512] or [b, 64, 512]
    y = y.to(params.device, non_blocking=True)

    # Calculate padding for height and width
    _, _, h, w = x.shape
    net_steps = params.unet_steps
    h_pad = max(2**net_steps - h % 2**net_steps, 0)
    w_pad = max(2**net_steps - w % 2**net_steps, 0)

    # Pad the input tensor
    x = F.pad(
        x, (w_pad // 2, w_pad // 2 + w_pad % 2, h_pad // 2, h_pad // 2 + h_pad % 2)
    )

    # Forward pass
    # with torch.cuda.amp.autocast():  # to use mixed precision (not working)
    y_pred = network(x[:, None])  # [b, 4, d, 64, 512] or [b, 4, 64, 512]

    # Crop the output tensor based on the padding
    crop_h_start = h_pad // 2
    crop_h_end = -(h_pad // 2 + h_pad % 2) if h_pad > 0 else None
    crop_w_start = w_pad // 2
    crop_w_end = -(w_pad // 2 + w_pad % 2) if w_pad > 0 else None
    y_pred = y_pred[:, :, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

    # Remove frames that must be ignored by the loss function
    if params.ignore_frames_loss != 0:
        y_pred = y_pred[:, :, params.ignore_frames_loss : -params.ignore_frames_loss]
        y = y[:, params.ignore_frames_loss : -params.ignore_frames_loss]

    # Handle specific loss functions
    if params.criterion == "dice_loss":
        # Set regions in pred where label is ignored to 0
        y_pred = y_pred * (y != 4)
        y = y * (y != 4)
    else:
        y = y.long()

    # Move criterion weights to GPU
    if hasattr(criterion, "weight") and not criterion.weight.is_cuda:
        criterion.weight = criterion.weight.to(params.device)
    if hasattr(criterion, "NLLLoss") and not criterion.NLLLoss.weight.is_cuda:
        criterion.NLLLoss.weight = criterion.NLLLoss.weight.to(params.device)

    # Compute loss
    loss = criterion(y_pred, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
        logger.debug(f"Current learning rate: {lr}")
        scheduler.step()

        new_lr = scheduler.get_last_lr()[0]
        if new_lr != lr:
            logger.info(f"Learning rate changed to {new_lr}")

    return {"loss": loss}


# Iterator (?) over the dataset
def mycycle(dataset_loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for sample in dataset_loader:
            yield sample


# _cycle = mycycle(dataset_loader)


def sampler(dataset_loader: torch.utils.data.DataLoader) -> dict:
    return next(mycycle(dataset_loader))  # (_cycle)


####################### Functions for data augmentation ########################


class TransformedSparkDataset(unet.TransformedDataset, SparkDataset):
    def __getitem__(self, idx: int) -> dict:
        value_dict = self.source_dataset[idx]  # dict with keys 'data', 'labels', etc.
        x = value_dict["data"]
        y = value_dict["labels"]

        # Apply transformations
        x, y = self.transform(x, y)

        # Assign transformed data to the dictionary
        value_dict["data"] = x
        value_dict["labels"] = y

        return value_dict


def random_flip(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly flip the input and target tensors along both horizontal and
    vertical axes.

    Args:
        x (Tensor): Input tensor (e.g., movie frames).
        y (Tensor): Target tensor (e.g., annotation mask).

    Returns:
        Tensor: Flipped input tensor.
        Tensor: Flipped target tensor.
    """

    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)
        y = y.flip(-1)

    if torch.rand(1).item() > 0.5:
        x = x.flip(-2)
        y = y.flip(-2)

    return x, y


def random_flip_noise(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random flips and noise to input and target tensors.

    Args:
        x (Tensor): Input tensor (e.g., movie frames).
        y (Tensor): Target tensor (e.g., annotation mask).

    Returns:
        Tensor: Transformed input tensor.
        Tensor: Transformed target tensor.
    """
    # Flip movie and annotation mask
    x, y = random_flip(x, y)

    # Add noise to movie with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of being normal or Poisson noise
        if torch.rand(1).item() > 0.5:
            noise = torch.normal(mean=0.0, std=1.0, size=x.shape)
            x = x + noise
        else:
            x = torch.poisson(x)  # not sure if this works...

    # Denoise input with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of gaussian filtering or median filtering
        # Convert to numpy array
        x_numpy = x.numpy()
        if torch.rand(1).item() > 0.5:
            # Apply gaussian filter
            x_numpy = ndi.gaussian_filter(x_numpy, sigma=1)
        else:
            x_numpy = ndi.median_filter(x_numpy, size=2)
        # Convert to tensor
        x = torch.as_tensor(x_numpy)

    return x, y


################## Functions related to UNet and loss weights ##################


def compute_class_weights(
    dataset: torch.utils.data.Dataset, weights: List[float] = []
) -> torch.Tensor:
    """
    Compute class weights for a dataset based on class frequencies.

    Args:
        dataset (Dataset): Dataset containing input-target pairs.
        w (list of floats): List of weights for each class.

    Returns:
        Tensor: Class weights as a tensor.
    """
    class_counts = torch.Tensor([0] * config.num_classes)

    with torch.no_grad():
        for _, y in dataset:
            for c in range(config.num_classes):
                class_counts[c] += torch.count_nonzero(y == c)

    total_samples = sum(class_counts)

    dataset_weights = torch.zeros(config.num_classes, dtype=torch.float32)

    if len(weights) == 0:
        weights = [1.0] * config.num_classes

    for c in range(config.num_classes):
        if class_counts[c] != 0:
            dataset_weights[c] = (
                weights[c] * total_samples / (config.num_classes * class_counts[c])
            )

    return dataset_weights


def compute_class_weights_instances(
    ys: List[np.ndarray], ys_events: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute class weights for event types (sparks, puffs, waves) based on
    annotation masks and event masks.
    Modified version of 'compute_class_weights' (using numpy arrays instead of a
    SparkDataset instance, number of events instead of number of pixels and not
    considering background class).

    Args:
        ys (list of numpy arrays): List of annotation masks with values between
            0 and 4.
        ys_events (list of numpy arrays): List of event masks with integer
        values.

    Returns:
        dict: Dictionary of weights for each event type.
    """
    # Initialize a dictionary to store event type counts
    event_type_counts = defaultdict(int)

    # Iterate through annotation masks and event masks
    for y, y_events in zip(ys, ys_events):
        for event_type, event_label in config.classes_dict.items():
            if event_label in [0, config.ignore_index]:
                continue

            # Create a binary mask for the event type
            class_mask = y == event_label

            # Combine the event mask with the class mask
            event_mask = y_events * class_mask

            # Count the unique event instances of the event type
            event_type_count = np.unique(event_mask).size - 1

            # Accumulate the count in the dictionary
            event_type_counts[event_type] += event_type_count

    # Calculate total count of event instances
    total_event_instances = sum(event_type_counts.values())

    # Calculate class weights as inverses of event type counts
    class_weights = {
        event_type: total_event_instances / (3 * count)
        for event_type, count in event_type_counts.items()
    }

    return class_weights


def weights_init(m: nn.Module) -> None:
    """
    Initialize weights of Conv2d and ConvTranspose2d layers using a specific method.

    Args:
        m (nn.Module): Neural network module.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        stdv = np.sqrt(2 / m.weight.size(1))
        m.weight.data.normal_(mean=float(m.weight), std=stdv)


############################### Inference tools ################################


@torch.no_grad()
def do_inference(
    network: nn.Module,
    params: TrainingConfig,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_loss: bool = False,
    inference_types: List[str] = [],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Given a trained network and a dataloader, run the data through the network
    and perform inference. The inference is done using different methods, which
    are specified in the inference_types list. The returned predictions are also
    cropped to the original duration of the movie.


    Args:
        network (nn.Module): The trained neural network model.
        params (TrainingConfig): Configuration parameters for the selected
            model/training.
        dataloader (torch.utils.data.DataLoader): Dataloader for input data.
        device (torch.device): The device on which to perform inference.
        compute_loss (bool, optional): If True, get the logit values instead of
            the probabilities. Defaults to False.
        inference_types (List[str], optional): List of inference types to apply.

    Returns:
        Dict[int, Dict[str, np.ndarray]]: A dictionary mapping movie IDs to
        dictionaries of inference type to predictions.
    """
    if len(inference_types) == 0:
        inference_types = [params.inference]

    # Loop over batches and store results and original duration for each movie
    preds_per_movie, original_duration_per_movie = run_dataloader_in_network(
        network=network,
        dataloader=dataloader,
        device=device,
        compute_loss=compute_loss,
    )

    # Loop over movies and compute the predictions for each inference type
    preds_all_movies_all_inferences = {}
    for movie_id in preds_per_movie.keys():
        # Initialize dict for storing the predictions for each inference type
        preds_all_inferences = {}

        # Concatenate the predictions for each movie using different types of inference
        for inference_type in inference_types:
            if inference_type == "overlap":
                # This is the default inference type used in training, it works
                # better than the others.

                half_overlap = get_half_overlap(
                    data_duration=params.data_duration,
                    data_stride=params.data_stride,
                    temporal_reduction=params.temporal_reduction,
                    num_channels=params.num_channels,
                )

                combined_preds = overlap_inference(
                    movie_preds=preds_per_movie[movie_id],
                    half_overlap=half_overlap,
                )
            else:
                # For the other inference types (average, gaussian, max), we
                # need to keep track of the predictions corresponding to each
                # frame in the movie.

                frame_predictions = from_chunks_to_frames(
                    prediction_chunks=preds_per_movie[movie_id],
                    data_stride=params.data_stride,
                )

                if inference_type == "average":
                    combined_preds = average_inference(frame_preds=frame_predictions)
                elif inference_type == "gaussian":
                    combined_preds = gaussian_inference(frame_preds=frame_predictions)
                elif inference_type == "max":
                    combined_preds = max_inference(frame_preds=frame_predictions)
                else:
                    raise ValueError(f"Unsupported inference type: {inference_type}")

            # Crop the predictions to the original duration of the movie
            combined_preds = remove_padding(
                preds=combined_preds,
                original_duration=original_duration_per_movie[movie_id],
            )

            # Convert the predictions to a numpy array
            combined_preds = combined_preds.cpu().numpy()

            preds_all_inferences[inference_type] = combined_preds

        # Store the predictions for the movie
        preds_all_movies_all_inferences[movie_id] = preds_all_inferences

    return preds_all_movies_all_inferences


@torch.no_grad()
def run_dataloader_in_network(
    network: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_loss: bool = False,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, int]]:
    """
    Run the data through the network using the specified dataloader.

    Args:
        network (nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): Dataloader for input data.
        device (torch.device): The device on which to perform inference.
        compute_loss (bool, optional): If True, get the logit values instead of
            the probabilities. Defaults to False.

    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, int]]: A tuple containing two
        dictionaries:
        - predictions_per_movie: A dictionary mapping movie IDs to lists of
            predictions (per chunk).
        - original_duration_per_movie: A dictionary mapping movie IDs to their
            original durations.
    """
    network.to(device)
    network.eval()

    # Loop over batches and store results and original duration for each movie
    predictions_per_movie = {}
    original_duration_per_movie = {}

    for batch in dataloader:
        batch_movie_ids = batch["movie_id"]
        batch_data = batch["data"].to(device, non_blocking=True)

        # Run the network on the batch
        batch_preds = network(batch_data[:, None])

        if not compute_loss:
            # If computing loss, use logit values; otherwise, compute
            # probabilities because it reduces the errors due to floating point
            # precision.
            batch_preds = torch.exp(batch_preds)

        # Add each movie_id in the batch to the dict if not already present
        for i, movie_id in enumerate(batch_movie_ids):
            movie_id = movie_id.item()
            if movie_id not in predictions_per_movie.keys():
                predictions_per_movie[movie_id] = []
                original_duration_per_movie[movie_id] = batch["original_duration"][
                    i
                ].item()

            predictions_per_movie[movie_id].append(batch_preds[i])

    return predictions_per_movie, original_duration_per_movie


def detect_nan_sample(x: torch.Tensor, y: torch.Tensor) -> None:
    """
    Detect NaN values in input tensors (x) and annotation tensors (y).

    Args:
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Annotation tensor.
    """
    if torch.isnan(x).any() or torch.isnan(y).any():
        if torch.isnan(x).any() or torch.isnan(y).any():
            logger.warning(
                "Detect NaN in network input (test): {}".format(torch.isnan(x).any())
            )
            logger.warning(
                "Detect NaN in network annotation (test): {}".format(
                    torch.isnan(y).any()
                )
            )


def from_chunks_to_frames(
    prediction_chunks: List[torch.Tensor], data_stride: int
) -> List[List[torch.Tensor]]:
    """
    Convert chunk predictions to frame predictions (for the inference types
    average, gaussian, and max we need to keep track of the predictions
    corresponding to each frame in the movie.)

    Args:
        prediction_chunks (List[torch.Tensor]): List of chunk predictions.
        data_stride (int): Stride between chunks.

    Returns:
        List[List[torch.Tensor]]: Frame predictions for each frame in the movie.
    """
    n_chunks = len(prediction_chunks)
    window_size = prediction_chunks[0].size(1)

    # Calculate the total number of frames in the movie
    total_frames = window_size + (n_chunks - 1) * data_stride

    # Initialize an empty list to store predictions associated with each frame
    frame_predictions: List[List[torch.Tensor]] = [[] for _ in range(total_frames)]

    # Loop through the list of prediction chunks
    for chunk_id, chunk in enumerate(prediction_chunks):
        # chunk has shape (n_classes, window_size, height, width)

        # Calculate the start index in the movie
        start_frame = chunk_id * data_stride

        # Loop through the frames in the chunk and append them to the list
        for frame_id in range(window_size):
            # Calculate the frame index in the movie
            frame_index = start_frame + frame_id

            # Get the prediction for the frame
            frame_prediction = chunk[:, frame_id]

            # Append the prediction to the list
            frame_predictions[frame_index].append(frame_prediction)

    return frame_predictions


def overlap_inference(
    movie_preds: List[torch.Tensor], half_overlap: int
) -> torch.Tensor:
    """
    Perform overlap inference on a list of chunk predictions.

    Args:
        movie_preds (List[torch.Tensor]): List of chunk predictions.
        half_overlap (int): The half overlap value.

    Returns:
        torch.Tensor: Combined predictions after overlap inference.
    """
    n_chunks = len(movie_preds)

    # Loop over chunks
    preds_list = []
    for i, chunk_pred in enumerate(movie_preds):
        # Define start and end of used frames in chunks
        start_mask = 0 if i == 0 else half_overlap
        end_mask = None if i + 1 == n_chunks else -half_overlap

        if chunk_pred.ndim == 4:
            preds_list.append(chunk_pred[:, start_mask:end_mask])
        else:
            preds_list.append(chunk_pred[:, None])

    # Concatenate the tensors of the predictions
    concatenated_preds = torch.cat(preds_list, dim=1)

    return concatenated_preds


def get_half_overlap(
    data_duration: int,
    data_stride: int,
    temporal_reduction: bool = False,
    num_channels: int = 0,
) -> int:
    """
    Calculate the half overlap value for overlap inference.

    Args:
        data_duration (int): Duration of the input data.
        data_stride (int): Stride between chunks.
        temporal_reduction (bool, optional): Whether temporal reduction is used.
        num_channels (int, optional): Number of channels in the input data (only
            used with temporal reduction).

    Returns:
        int: The half overlap value.
    """
    # Check and set up overlap inference
    if (data_duration - data_stride) % 2 == 0:
        half_overlap = (data_duration - data_stride) // 2
    else:
        raise ValueError(
            "data_duration - data_stride must be even for overlap inference."
        )

    # Adapt half_overlap duration if using temporal reduction
    if temporal_reduction:
        if half_overlap % num_channels == 0:
            half_overlap_mask = half_overlap // num_channels
        else:
            raise ValueError(
                "half_overlap must be divisible by num_channels for temporal reduction."
            )
    else:
        half_overlap_mask = half_overlap

    return half_overlap_mask


def average_inference(frame_preds: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Combine frame predictions using the average method.

    Args:
        frame_preds (List[List[torch.Tensor]]): List of frame predictions.

    Returns:
        torch.Tensor: Averaged frame predictions.
    """
    averaged_preds = [torch.stack(p).mean(dim=0) for p in frame_preds]

    # Convert the list to a tensor
    averaged_preds = torch.stack(averaged_preds)
    averaged_preds = averaged_preds.swapaxes(0, 1)

    return averaged_preds


def gaussian_inference(frame_preds: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Combine frame predictions using the Gaussian method.

    Args:
        frame_preds (List[List[torch.Tensor]]): List of frame predictions.

    Returns:
        torch.Tensor: Gaussian-combined frame predictions.
    """
    gaussian_preds = [
        (torch.stack(p) * gaussian(len(p)).view(-1, 1, 1, 1)).sum(dim=0)
        for p in frame_preds
    ]

    # Convert the list to a tensor
    gaussian_preds = torch.stack(gaussian_preds)
    gaussian_preds = gaussian_preds.swapaxes(0, 1)

    return gaussian_preds


def gaussian(n_chunks: int) -> torch.Tensor:
    """
    Generate a normalized Gaussian function with standard deviation 3.

    Args:
        n_chunks (int): Number of data points in the Gaussian function.

    Returns:
        torch.Tensor: Normalized Gaussian tensor.
    """

    sigma = 3
    x = np.linspace(-10, 10, n_chunks)

    c = np.sqrt(2 * np.pi)
    res = np.exp(-0.5 * (x / sigma) ** 2) / sigma / c
    return torch.as_tensor(res / res.sum())


def max_inference(frame_preds: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Combine frame predictions using the max method, keeping the class with the
    highest probability per pixel.

    Args:
        frame_preds (List[List[torch.Tensor]]): List of frame predictions.

    Returns:
        torch.Tensor: Max-combined frame predictions.
    """
    max_preds = []
    for p in frame_preds:
        p = torch.stack(p)

        # Get the max probability for each pixel
        max_ids = p.max(dim=1)[0].max(dim=0)[1]
        # View as a list of pixels and expand to frame size
        max_ids = max_ids.view(-1)[None, None, :].expand(p.size(0), p.size(1), -1)

        # View p as a list of pixels and gather predictions for each pixel
        # according to the chunk with the highest probability
        max_preds.append(
            p.view(p.size(0), p.size(1), -1)
            .gather(dim=0, index=max_ids)
            .view(*p.size())[0]
        )

    # Convert the list to a tensor
    max_preds = torch.stack(max_preds)
    max_preds = max_preds.swapaxes(0, 1)

    return max_preds


# function to run a test sample (i.e., a test dataset) in the UNet
@torch.no_grad()
def get_raw_preds_dict(
    model: torch.nn.Module,
    params: TrainingConfig,
    test_dataset: torch.utils.data.Dataset,
    criterion: Optional[torch.nn.Module] = None,
    inference_types: Optional[List[str]] = None,
):  # TODO
    """
    Given a trained model and a test sample (i.e., a test dataset), run the
    sample in the model and return the predictions.

    Args:
        model (torch.nn.Module): The trained neural network model.
        test_dataset (torch.utils.data.Dataset): The test dataset containing the
            sample(s).
        params (TrainingConfig): A TrainingConfig containing various parameters.
        criterion (torch.nn.Module, optional): If provided, the loss criterion
            for computing loss.
        inference_types (list of str, optional): List of inference types to use,
            or None to use the default type.

    Returns:
    TODO
    """
    if inference_types is None:
        assert params.inference in [
            "overlap",
            "average",
            "gaussian",
            "max",
        ], f"inference type '{params.inference}' not implemented yet"
        inference_types = [params.inference]

    else:
        assert all(
            i in ["overlap", "average", "gaussian", "max"] for i in inference_types
        ), "Unsupported inference type."

    # Create a dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Run movie in the network and perform inference
    preds = do_inference(
        network=model,
        params=params,
        test_dataloader=test_dataloader,
        device=params.device,
        compute_loss=True if criterion is not None else False,
        inference_types=inference_types,
        return_dict=return_dict,
    )

    # Get original movie xs and annotations ys
    xs = test_dataset.data[0]
    if test_dataset.gt_available:
        ys = test_dataset.annotations[0]

    # Remove padded frames
    pad = test_dataset.pad
    if pad > 0:
        start_pad = pad // 2
        end_pad = -(pad // 2 + pad % 2)
        xs = xs[start_pad:end_pad]

        if test_dataset.temporal_reduction:
            start_pad = start_pad // test_dataset.num_channels
            end_pad = end_pad // test_dataset.num_channels

        if params.nn_architecture != "unet_lstm":
            if test_dataset.gt_available:
                ys = ys[start_pad:end_pad]
            if len(inference_types) == 1:
                if not return_dict:
                    preds = preds[:, start_pad:end_pad]
                else:
                    preds = {
                        event_type: pred[start_pad:end_pad]
                        for event_type, pred in preds.items()
                    }
            else:
                if not return_dict:
                    preds = {i: p[:, start_pad:end_pad] for i, p in preds.items()}
                else:
                    for i, preds_dict in preds.items():
                        preds[i] = {
                            event_type: pred[start_pad:end_pad]
                            for event_type, pred in preds_dict.items()
                        }

        else:
            raise NotImplementedError

    # If original sample was shorter than the current movie duration,
    # remove additional padded frames
    movie_duration = test_dataset.movie_duration
    if movie_duration < xs.shape[0]:
        pad = xs.shape[0] - movie_duration
        start_pad = pad // 2
        end_pad = -(pad // 2 + pad % 2)
        xs = xs[start_pad:end_pad]

        if test_dataset.temporal_reduction:
            start_pad = start_pad // test_dataset.num_channels
            end_pad = end_pad // test_dataset.num_channels

        if ys is not None:
            ys = ys[start_pad:end_pad]

        if len(inference_types) == 1:
            if not return_dict:
                preds = preds[:, start_pad:end_pad]
            else:
                preds = {
                    event_type: pred[start_pad:end_pad]
                    for event_type, pred in preds.items()
                }
        else:
            if not return_dict:
                preds = {i: p[:, start_pad:end_pad] for i, p in preds.items()}
            else:
                for i, preds_dict in preds.items():
                    preds[i] = {
                        event_type: pred[start_pad:end_pad]
                        for event_type, pred in preds_dict.items()
                    }

    if criterion is not None:
        assert ys is not None, "Cannot compute loss if annotations are not available."

        if ys.ndim == 3:
            if len(inference_types) == 1 and not return_dict:
                preds_loss = preds[
                    :, test_dataset.ignore_frames : -test_dataset.ignore_frames
                ]
            else:
                raise NotImplementedError
                # Still need to adapt code to compute loss for list of inference
                # types, however usually loss should be computed only during
                # training, and therefore inference_types should be None.
                # Similarly, return_dict should be False.

            ys_loss = ys[test_dataset.ignore_frames : -test_dataset.ignore_frames]
        else:
            raise NotImplementedError

        if params.criterion == "dice_loss":
            # set regions in pred where label is ignored to 0
            preds_loss = preds_loss * (ys_loss != 4)
            ys_loss = ys_loss * (ys_loss != 4)
        else:
            ys_loss = ys_loss.long()[None, :]
            preds_loss = preds_loss[None, :]

        # Move criterion weights to cpu
        if hasattr(criterion, "weight") and criterion.weight.is_cuda:
            criterion.weight = criterion.weight.cpu()
        if hasattr(criterion, "NLLLoss") and criterion.NLLLoss.weight.is_cuda:
            criterion.NLLLoss.weight = criterion.NLLLoss.weight.cpu()

        loss = criterion(preds_loss, ys_loss).item()
        return xs.numpy(), ys.numpy(), preds.numpy(), loss

    else:
        if len(inference_types) == 1:
            if not return_dict:
                preds = preds.numpy()
            else:
                preds = {event_type: pred.numpy() for event_type, pred in preds.items()}
        else:
            if not return_dict:
                preds = {i: p.numpy() for i, p in preds.items()}
            else:
                for i, preds_dict in preds.items():
                    preds[i] = {
                        event_type: pred.numpy()
                        for event_type, pred in preds_dict.items()
                    }

    return xs.numpy(), ys.numpy(), preds if ys is not None else xs.numpy(), preds


@torch.no_grad()
def get_preds_from_path(  # TODO: vedere se si può eliminare e tenere solo get_preds
    model: nn.Module,
    params: TrainingConfig,
    movie_path: str,
    output_dir: Optional[str] = None,
) -> [Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function to get predictions from a movie path.

    Args:
    - model (torch.nn.Module): The trained neural network model.
    - params (TrainingConfig): A TrainingConfig containing various parameters.
    - movie_path: Path to the movie.
    - return_dict (bool, optional): Whether to return a dictionary with
        inference type as key and predictions as value, or a single tensor of
        predictions. Defaults to False.
    - output_dir: If not None, save raw predictions on disk.

    Returns:
    - If return_dict is True, return a dictionary with keys 'sparks', 'puffs',
        'waves'; else return a tuple of numpy arrays with integral values for
        classes and instances.
    """

    ### Get sample as dataset ###
    sample_dataset = SparkDatasetInference(
        sample_path=movie_path,
        params=params,
        # resampling=False, # It could be implemented later
        # resampling_rate=150,
    )

    ### Run sample in UNet ###
    input_movie, preds_dict = get_raw_preds_dict(
        model=model,
        test_dataset=sample_dataset,
        params=params,
        inference_types=None,
        return_dict=True,
    )

    ### Get processed output ###

    # Get predicted segmentation and event instances
    preds_instances, preds_segmentation, _ = process_raw_predictions(
        raw_preds_dict=preds_dict,
        input_movie=input_movie,
        training_mode=False,
        debug=False,
    )
    # preds_instances and preds_segmentations are dictionaries
    # with keys 'sparks', 'puffs', 'waves'.

    # Save raw preds on disk ### I don't know if this is necessary
    if output_dir is not None:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        write_videos_on_disk(
            training_name=None,
            video_name=sample_dataset.video_name,
            path=output_dir,
            preds=[
                None,
                preds_dict["sparks"],
                preds_dict["waves"],
                preds_dict["puffs"],
            ],
            ys=None,
        )

    if return_dict:
        return preds_segmentation, preds_instances

    else:
        # Get integral values for classes and instances
        preds_segmentation = preds_dict_to_mask(preds_segmentation)
        preds_instances = sum(preds_instances.values())
        # Instances already have different IDs

        return preds_segmentation, preds_instances


################################ Test function #################################


# TODO: ormai anche per i test datasets si usa la stessa classe che per il
# training, quindi devo aggiornare il loop che dalla lista dei datasets
# andrà a far passare direttamente i vari video contenuti nell'unico dataset.
def test_function(  # da aggiornare in base alla nuova definizione dei datasets
    network: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    params: TrainingConfig,
    testing_datasets: List[SparkDataset],
    training_name: str,
    output_dir: str,
    training_mode: bool = True,
    debug: bool = False,
) -> None:
    """
    Validate UNet during training.
    Output segmentation is computed using argmax values and Otsu threshold (to
    remove artifacts & avoid using thresholds).
    Removing small events prior metrics computation as well.

    network:            the model being trained
    device:             current device
    criterion:          loss function to be computed on the validation set
    testing_datasets:   list of SparkDataset instances
    training_name:      training name used to save predictions on disk
    output_dir:         directory where the predicted movies are saved
    training_mode:      bool, if True, separate events using a simpler algorithm
    """

    network.eval()

    # Initialize dicts that will contain the results
    tot_preds = {"sparks": 0, "puffs": 0, "waves": 0}
    tp_preds = {"sparks": 0, "puffs": 0, "waves": 0}
    ignored_preds = {"sparks": 0, "puffs": 0, "waves": 0}
    unlabeled_preds = {"sparks": 0, "puffs": 0, "waves": 0}
    tot_ys = {"sparks": 0, "puffs": 0, "waves": 0}
    tp_ys = {"sparks": 0, "puffs": 0, "waves": 0}
    undetected_ys = {"sparks": 0, "puffs": 0, "waves": 0}

    # Compute loss on all samples
    sum_loss = 0.0

    # Concatenate annotations and preds to compute segmentation-based metrics
    ys_concat = []
    preds_concat = []

    for test_dataset in testing_datasets:
        ########################## Run sample in UNet ##########################

        start = time.time()

        # Get video name
        video_name = test_dataset.video_name
        logger.debug(f"Testing function: running sample {video_name} in UNet")

        # Run sample in UNet, returns a list [bg, sparks, waves, puffs]
        xs, ys, preds, loss = get_raw_preds_dict(
            model=network,
            test_dataset=test_dataset,
            device=device,
            criterion=criterion,
            batch_size=params.batch_size,
            detect_nan=False,
        )

        # Sum up losses of each sample
        sum_loss += loss

        # Compute exp of predictions
        preds = np.exp(preds)

        # Save raw predictions as .tif videos
        write_videos_on_disk(
            xs=xs,
            ys=ys,
            preds=preds,
            training_name=training_name,
            video_name=video_name,
            path=output_dir,
        )

        logger.debug(
            f"Time to run sample {video_name} in UNet: {time.time() - start:.2f} s"
        )

        ####################### Re-organise annotations ########################

        start = time.time()
        logger.debug("Testing function: re-organising annotations")

        # ys_instances is a dict with classified event instances, for each class
        ys_instances = masks_to_instances_dict(
            instances_mask=test_dataset.events, labels_mask=ys, shift_ids=True
        )

        # Remove ignored events entry from ys_instances
        ys_instances.pop("ignore", None)

        # Get ignored pixels mask
        # TODO: remove ignored frames as well?
        ignore_mask = np.where(ys == config.ignore_index, 1, 0)

        logger.debug(f"Time to re-organise annotations: {time.time() - start:.2f} s")

        ######################### Get processed output #########################

        logger.debug(
            "Testing function: getting processed output (segmentation and instances)"
        )

        # Get predicted segmentation and event instances
        raw_preds_dict = {
            "sparks": preds[1],
            "puffs": preds[3],
            "waves": preds[2],
        }
        preds_instances, preds_segmentation, _ = process_raw_predictions(
            raw_preds_dict=raw_preds_dict,
            input_movie=xs,
            training_mode=training_mode,
            debug=debug,
        )

        ##################### Stack ys and segmented preds #####################

        # Stack annotations and remove marginal frames
        ys_concat.append(
            trim_and_pad_video(video=ys, n_margin_frames=params.ignore_frames_loss)
        )

        # Stack preds and remove marginal frames
        temp_preds = np.zeros_like(ys)
        for event_type, event_label in config.classes_dict.items():
            if event_type == "ignore":
                continue
            temp_preds += event_label * preds_segmentation[event_type]
        preds_concat.append(
            trim_and_pad_video(
                video=temp_preds, n_margin_frames=params.ignore_frames_loss
            )
        )

        logger.debug(f"Time to process predictions: {time.time() - start:.2f} s")

        ############### Compute pairwise scores (based on IoMin) ###############

        start = time.time()

        if debug:
            n_ys_events = max(
                [
                    np.max(ys_instances[event_type])
                    for event_type in config.classes_dict.keys()
                    if event_type != "ignore"
                ]
            )

            n_preds_events = max(
                [
                    np.max(preds_instances[event_type])
                    for event_type in config.classes_dict.keys()
                    if event_type != "ignore"
                ]
            )
            logger.debug(
                f"Testing function: computing pairwise scores between {n_ys_events} annotated events and {n_preds_events} predicted events"
            )

        iomin_scores = get_score_matrix(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            ignore_mask=None,
            score="iomin",
        )

        logger.debug(f"Time to compute pairwise scores: {time.time() - start:.2f} s")

        ####################### Get matches summary #######################

        start = time.time()

        logger.debug("Testing function: getting matches summary")

        matched_ys_ids, matched_preds_ids = get_matches_summary(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            scores=iomin_scores,
            ignore_mask=ignore_mask,
        )

        # Count number of categorized events that are necessary for the metrics
        for ca_event in config.classes_dict.keys():
            if ca_event == "ignore":
                continue
            tot_preds[ca_event] += len(matched_preds_ids[ca_event]["tot"])
            tp_preds[ca_event] += len(matched_preds_ids[ca_event]["tp"])
            ignored_preds[ca_event] += len(matched_preds_ids[ca_event]["ignored"])
            unlabeled_preds[ca_event] += len(matched_preds_ids[ca_event]["unlabeled"])

            tot_ys[ca_event] += len(matched_ys_ids[ca_event]["tot"])
            tp_ys[ca_event] += len(matched_ys_ids[ca_event]["tp"])
            undetected_ys[ca_event] += len(matched_ys_ids[ca_event]["undetected"])

        logger.debug(f"Time to get matches summary: {time.time() - start:.2f} s")

    ############################## Reduce metrics ##############################

    start = time.time()

    logger.debug("Testing function: reducing metrics")

    metrics = {}

    # Compute average validation loss
    metrics["validation_loss"] = sum_loss / len(testing_datasets)

    ##################### Compute instances-based metrics ######################

    """
    Metrics that can be computed (event instances):
    - Confusion matrix
    - Precision & recall (TODO)
    - F-score (e.g. beta = 0.5,1,2) (TODO)
    (- Matthews correlation coefficient (MCC))??? (TODO)
    """

    # Get confusion matrix of all summed events
    # metrics["events_confusion_matrix"] = sum(confusion_matrix.values())

    # Get other metrics (precision, recall, % correctly classified, % detected)
    metrics_all = get_metrics_from_summary(
        tot_preds=tot_preds,
        tp_preds=tp_preds,
        ignored_preds=ignored_preds,
        unlabeled_preds=unlabeled_preds,
        tot_ys=tot_ys,
        tp_ys=tp_ys,
        undetected_ys=undetected_ys,
    )

    metrics.update(metrics_all)

    #################### Compute segmentation-based metrics ####################

    """
    Metrics that can be computed (raw sparks, puffs, waves):
    - Jaccard index (IoU)
    - Dice score (TODO)
    - Precision & recall (TODO)
    - F-score (e.g. beta = 0.5,1,2) (TODO)
    - Accuracy (biased since background is predominant) (TODO)
    - Matthews correlation coefficient (MCC) (TODO)
    - Confusion matrix
    """

    # Concatenate annotations and preds
    ys_concat = np.concatenate(ys_concat, axis=0)
    preds_concat = np.concatenate(preds_concat, axis=0)

    # Concatenate ignore masks
    ignore_concat = ys_concat == 4

    for event_type, event_label in config.classes_dict.items():
        if event_type == "ignore":
            continue
        class_preds = preds_concat == event_label
        class_ys = ys_concat == event_label

        metrics["segmentation/" + event_type + "_IoU"] = compute_iou(
            ys_roi=class_ys, preds_roi=class_preds, ignore_mask=ignore_concat
        )

    # Get average IoU across all classes
    metrics["segmentation/average_IoU"] = np.mean(
        [
            metrics["segmentation/" + event_type + "_IoU"]
            for event_type in config.classes_dict.keys()
            if event_type != "ignore"
        ]
    )

    # Compute confusion matrix
    metrics["segmentation_confusion_matrix"] = sk_confusion_matrix(
        y_true=ys_concat.flatten(), y_pred=preds_concat.flatten(), labels=[0, 1, 2, 3]
    )

    logger.debug(f"Time to reduce metrics: {time.time() - start:.2f} s")

    return metrics


########################### Custom training manager ############################


class MyTrainingManager(unet.TrainingManager):
    """
    Custom training manager for deep learning training.
    """

    def run_validation(self, wandb_log: bool = False) -> None:
        """
        Run validation on the network and log the results.

        Args:
            wandb_log (bool, optional): Whether to log results to WandB.
                    Defaults to False.
        """

        if self.test_function is None:
            return

        logger.info(f"Validating network at iteration {self.iter}...")

        test_output = self.test_function(self.iter)

        if wandb_log:
            self.log_test_metrics_to_wandb(test_output)

        self.log_metrics(test_output, "Metrics:")

        _write_results(self.summary, "testing", test_output, self.iter)

    def train(
        self,
        num_iters: int,
        print_every: int = 0,
        maxtime: float = np.inf,
        wandb_log: bool = False,
    ) -> None:
        """
        Train the deep learning model.

        Args:
            num_iters (int): Number of training iterations.
            print_every (int, optional): Print training information every
                'print_every' iterations. Defaults to 0.
            maxtime (float, optional): Maximum training time in seconds.
                Defaults to np.inf.
            wandb_log (bool, optional): Whether to log results to WandB.
                Defaults to False.
        """
        tic = time.process_time()
        time_elapsed = 0

        loss_sum = 0  # for wandb

        for _ in range(num_iters):
            step_output = self.training_step(self.iter)

            if wandb_log:
                loss_sum += step_output["loss"].item()

            time_elapsed = time.process_time() - tic

            if print_every and self.iter % print_every == 0:
                self.log_training_info(step_output, time_elapsed)

                # Log to wandb
                if wandb_log:
                    wandb.log(
                        {
                            "U-Net training loss": loss_sum / print_every
                            if self.iter > 0
                            else loss_sum
                        },
                        step=self.iter,
                    )
                    loss_sum = 0

            self.iter += 1

            # Validation
            if self.test_every and self.iter % self.test_every == 0:
                self.run_validation(wandb_log=wandb_log)

            # Plot
            if (
                self.plot_function
                and self.plot_every
                and self.iter % self.plot_every == 0
            ):
                self.plot_function(self.iter, self.summary)

            # Save model and solver
            if self.save_every and self.iter % self.save_every == 0:
                self.save()

            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break

    def log_test_metrics_to_wandb(self, test_output: Dict[str, float]) -> None:
        """
        Log test metrics to WandB.

        Args:
            test_output (dict): Dictionary of test metrics.
        """
        wandb.log(
            {m: val for m, val in test_output.items() if "confusion_matrix" not in m},
            step=self.iter,
        )

    def log_metrics(
        self, metrics_dict: Dict[str, float], header: str = "Metrics:"
    ) -> None:
        """
        Log metrics to the logger.

        Args:
            metrics_dict (dict): Dictionary of metrics.
            header (str, optional): Header for the metrics section.
                Defaults to "Metrics:".
        """
        logger.info(header)
        for metric_name, metric_value in metrics_dict.items():
            if "confusion_matrix" in metric_name:
                with np.printoptions(precision=0, suppress=True):
                    logger.info(f"\t{metric_name}:\n{metric_value:}")
            else:
                logger.info(f"\t{metric_name}: {metric_value:.4g}")

    def log_training_info(
        self, step_output: Dict[str, torch.Tensor], time_elapsed: float
    ) -> None:
        """
        Log training information to the logger.

        Args:
            step_output (dict): Output from the training step.
            time_elapsed (float): Elapsed time for the current iteration.
        """
        # Register data
        _write_results(self.summary, "training", step_output, self.iter)
        loss = step_output["loss"].item()  # Move loss value to cpu

        # Check for nan
        if np.any(np.isnan(loss)):
            logger.error("Last loss is nan! Training diverged!")
            return

        # Log training loss
        logger.info(f"Iteration {self.iter}...")
        logger.info(f"\tTraining loss: {loss:.4g}")
        logger.info(f"\tTime elapsed: {time_elapsed:.2f}s")
