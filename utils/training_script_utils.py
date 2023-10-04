"""
Functions used in training.py and inference.py to initialize parameters,
datasets, model and loss function, to make the code more readable.

Functions:
    init_config_file_path: Load configuration file.
    init_training_dataset: Initialize training dataset.
    init_testing_dataset: Initialize testing dataset.
    init_model: Initialize deep learning model.
    init_criterion: Initialize loss function.

Author: Prisca Dotti
Last modified: 27.09.2023
"""
import argparse
import logging
import os

from torch import nn

import models.UNet as unet
import models.unetOpenAI as unet_openai
from config import config
from data.datasets import SparkDataset, SparkDatasetLSTM
from models.architectures import TempRedUNet, UNetConvLSTM
from models.new_unet import UNet
from utils.custom_losses import (
    Dice_CELoss,
    FocalLoss,
    LovaszSoftmax,
    LovaszSoftmax3d,
    SumFocalLovasz,
    mySoftDiceLoss,
)
from utils.training_inference_tools import (
    compute_class_weights,
    random_flip,
    random_flip_noise,
)

__all__ = [
    "init_config_file_path",
    "init_training_dataset",
    "init_testing_dataset",
    "init_model",
    "init_criterion",
]

logger = logging.getLogger(__name__)


def init_config_file_path():
    """
    Initialize the path to the configuration file based on the provided argument or
    command line input.

    Returns:
        str: The full path to the configuration file.
    """

    # Try to get the configuration file name from the command line
    try:
        parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")
        parser.add_argument(
            "config", type=str, help="Input config file, used to configure training"
        )
        args = parser.parse_args()

        return args.config
    except Exception as e:
        logger.error(
            "No configuration file provided. Please provide a configuration file."
        )
        exit()


def init_training_dataset(params, train_sample_ids, print_dataset_info=True):
    """
    Initialize the training dataset based on provided parameters and sample IDs.

    Args:
        params (TrainingConfig): TrainingConfig instance containing configuration
            parameters.
        train_sample_ids (list): List of sample IDs for training.
        print_dataset_info (bool, optional): Whether to print dataset information.
            Default is True.

    Returns:
        SparkDataset or SparkDatasetLSTM: Initialized training dataset.
    """

    # Define dataset path
    dataset_path = os.path.realpath(f"{config.basedir}/{params.relative_path}")

    assert os.path.isdir(dataset_path), f'"{dataset_path}" is not a directory'

    dataset_args = {
        "base_path": dataset_path,
        "sample_ids": train_sample_ids,
        "testing": False,
        "smoothing": params.data_smoothing,
        "step": params.data_step,
        "duration": params.data_duration,
        "remove_background": params.remove_background,
        "temporal_reduction": params.temporal_reduction,
        "num_channels": params.num_channels,
        "normalize_video": params.norm_video,
        "only_sparks": params.only_sparks,
        "sparks_type": params.sparks_type,
        "ignore_index": config.ignore_index,
        "inference": None,
    }

    if params.nn_architecture == "unet_lstm":
        dataset = SparkDatasetLSTM(**dataset_args)
    elif params.nn_architecture in ["pablos_unet", "github_unet", "openai_unet"]:
        dataset = SparkDataset(**dataset_args)
    else:
        logger.error(f"{params.nn_architecture} is not a valid nn architecture.")
        exit()

    # Apply transforms based on noise_data_augmentation setting
    # (transforms are applied when getting a sample from the dataset)
    transforms = random_flip_noise if params.noise_data_augmentation else random_flip
    dataset = unet.TransformedDataset(dataset, transforms)

    if print_dataset_info:
        logger.info(f"Samples in training dataset: {len(dataset)}")

    return dataset


def init_testing_dataset(params, test_sample_ids, print_dataset_info=True):
    """
    Initialize the testing dataset based on provided parameters and sample IDs.

    Args:
        params (TrainingConfig): TrainingConfig instance containing configuration
            parameters.
        test_sample_ids (list): List of sample IDs for testing.
        print_dataset_info (bool, optional): Whether to print dataset information.
            Default is True.

    Returns:
        list: List of initialized testing datasets.
    """

    # Define dataset path
    dataset_path = os.path.realpath(f"{config.basedir}/{params.relative_path}")

    assert os.path.isdir(dataset_path), f'"{dataset_path}" is not a directory'

    testing_datasets = []

    dataset_args = {
        "base_path": dataset_path,
        "sample_ids": test_sample_ids,
        "testing": True,
        "smoothing": params.data_smoothing,
        "step": params.testing_data_step,
        "duration": params.data_duration,
        "remove_background": params.remove_background,
        "temporal_reduction": params.temporal_reduction,
        "num_channels": params.num_channels,
        "normalize_video": params.norm_video,
        "only_sparks": params.only_sparks,
        "sparks_type": params.sparks_type,
        "ignore_frames": params.ignore_frames_loss,
        "ignore_index": config.ignore_index,
        "inference": params.inference,
    }

    if params.nn_architecture == "unet_lstm":
        DatasetClass = SparkDatasetLSTM
    elif params.nn_architecture in ["pablos_unet", "github_unet", "openai_unet"]:
        DatasetClass = SparkDataset
    else:
        logger.error(f"{params.nn_architecture} is not a valid nn architecture.")
        exit()

    for sample_id in test_sample_ids:
        dataset = DatasetClass(**dataset_args, sample_ids=[sample_id])
        testing_datasets.append(dataset)

        if print_dataset_info:
            logger.info(
                f"Testing dataset for sample {sample_id} contains {len(dataset)} samples"
            )

    return testing_datasets


def init_model(params):
    """
    Initialize the deep learning model based on the specified architecture in params.

    Args:
        params (TrainingConfig): TrainingConfig instance containing configuration
            parameters.

    Returns:
        torch.nn.Module: Initialized deep learning model.
    """
    if params.nn_architecture == "pablos_unet":
        batch_norm = {"batch": True, "none": False}

        unet_config = unet.UNetConfig(
            steps=params.unet_steps,
            first_layer_channels=params.first_layer_channels,
            num_classes=config.num_classes,
            ndims=config.ndims,
            dilation=params.dilation,
            border_mode=params.border_mode,
            batch_normalization=batch_norm[params.batch_normalization],
            num_input_channels=params.num_channels,
        )

        if not params.temporal_reduction:
            network = unet.UNetClassifier(unet_config)
        else:
            assert (
                params.data_duration % params.num_channels == 0
            ), "using temporal reduction chunks_duration must be a multiple of num_channels"
            network = TempRedUNet(unet_config)

    elif params.nn_architecture == "github_unet":
        network = UNet(
            in_channels=params.num_channels,
            out_channels=config.num_classes,
            n_blocks=params.unet_steps + 1,
            start_filts=params.first_layer_channels,
            up_mode=params.up_mode,
            # up_mode = 'transpose', # TESTARE DIVERSE POSSIBILTÀ
            # up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
            merge_mode="concat",  # Default, dicono che funziona meglio
            # planar_blocks=(0,), # magari capire cos'è e testarlo ??
            activation="relu",
            # Penso che nell'implementazione di Pablo normalization è 'none'
            normalization=params.batch_normalization,
            attention=params.attention,  # magari da testare con 'True' ??
            # full_norm=False,  # Uncomment to restore old sparse normalization scheme
            dim=config.ndims,
            # 'valid' ha dei vantaggi a quanto pare...
            conv_mode=params.border_mode,
        )

    elif params.nn_architecture == "unet_lstm":
        batch_norm = {"batch": True, "none": False}
        config.ndims = 2  # convolutions applied to single frames

        unet_config = unet.UNetConfig(
            steps=params.unet_steps,
            first_layer_channels=params.first_layer_channels,
            num_classes=config.num_classes,
            ndims=config.ndims,
            dilation=params.dilation,
            border_mode=params.border_mode,
            batch_normalization=batch_norm[params.batch_normalization],
            # num_input_channels=params.data_duration, # frames seen as modalities
            num_input_channels=params.num_channels,
        )

        network = UNetConvLSTM(unet_config, bidirectional=params.bidirectional)

    elif params.nn_architecture == "openai_unet":
        if params.unet_steps == 4:
            channel_mult = (1, 2, 4, 8)
        elif params.unet_steps == 3:
            channel_mult = (1, 2, 4)
        elif params.unet_steps == 2:
            channel_mult = (1, 2)
        else:
            logger.error(f"{params.unet_steps} is not a valid number of unet steps.")
            exit()

        network = unet_openai.UNetModel(
            # image_size=x.shape[1:],
            in_channels=params.num_channels,
            model_channels=params.first_layer_channels,
            out_channels=config.num_classes,
            channel_mult=channel_mult,
            num_res_blocks=params.num_res_blocks,
            attention_resolutions=[],
            # attention_resolutions=[8], # bottom layer
            dropout=0.0,
            dims=config.ndims,
            # use_checkpoint=True
        )

    return network


def init_criterion(params, dataset):
    """
    Initialize the loss function based on the specified criterion in params.

    Args:
        params (TrainingConfig): TrainingConfig instance containing configuration
            parameters.
        dataset (torch.utils.data.Dataset): The training dataset.

    Returns:
        torch.nn.Module: Initialized loss function.
    """
    if params.criterion == "nll_loss":
        # Compute class weights
        class_weights = compute_class_weights(dataset)
        logger.info(
            "Using class weights: {}".format(
                ", ".join(str(w.item()) for w in class_weights)
            )
        )

        # Initialize the loss function
        criterion = nn.NLLLoss(
            ignore_index=config.ignore_index,
            weight=class_weights  # .to(
            # params.device, non_blocking=True)
        )

    elif params.criterion == "focal_loss":
        # Compute class weights
        class_weights = compute_class_weights(dataset)
        logger.info(
            "Using class weights: {}".format(
                ", ".join(str(w.item()) for w in class_weights)
            )
        )

        # Initialize the loss function
        criterion = FocalLoss(
            reduction="mean",
            ignore_index=config.ignore_index,
            alpha=class_weights,
            gamma=params.gamma,
        )

    elif params.criterion == "lovasz_softmax":
        # Initialize the loss function
        if params.nn_architecture in ["pablos_unet", "github_unet", "openai_unet"]:
            criterion = LovaszSoftmax3d(
                classes="present", per_image=False, ignore=config.ignore_index
            )
        elif params.nn_architecture == "unet_lstm":
            criterion = LovaszSoftmax(
                classes="present", per_image=True, ignore=config.ignore_index
            )

    elif params.criterion == "sum_losses":
        # Compute class weights
        class_weights = compute_class_weights(dataset)
        logger.info(
            "Using class weights: {}".format(
                ", ".join(str(w.item()) for w in class_weights)
            )
        )

        # Initialize the loss function
        criterion = SumFocalLovasz(
            classes="present",
            per_image=False,
            ignore=config.ignore_index,
            alpha=class_weights,
            gamma=params["gamma"],
            reduction="mean",
            w=params["w"],
        )

    elif params.criterion == "dice_loss":
        # Initialize the loss function
        softmax = nn.Softmax(dim=1)
        criterion = mySoftDiceLoss(apply_nonlin=softmax, batch_dice=True, do_bg=False)

    elif params.criterion == "dice_nll_loss":
        # Compute class weights
        class_weights = compute_class_weights(dataset)
        logger.info(
            "Using class weights: {}".format(
                ", ".join(str(w.item()) for w in class_weights)
            )
        )

        # Initialize the loss function
        criterion = Dice_CELoss(
            ignore_index=config.ignore_index,
            weight=class_weights,
        )

    else:
        logger.error(f"{params.criterion} is not a valid criterion.")
        exit()

    return criterion
