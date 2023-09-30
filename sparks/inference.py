"""
Load a saved UNet model at given epochs and save its predictions in the
folder `trainings_validation`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the
          results.

Author: Prisca Dotti
Last modified: 28.09.2023
"""

import logging
import os

import numpy as np
import torch
from architectures import TempRedUNet
from datasets import SparkDataset
from in_out_tools import write_videos_on_disk
from torch import nn
from training_inference_tools import get_preds

import unet

from config import config, TrainingConfig

logger = logging.getLogger(__name__)
config.verbosity = 3  # To get debug messages

####################### Get training-specific parameters #######################

# Initialize training-specific parameters
config_path = os.path.join("config_files", "config_final_model.ini")
params = TrainingConfig(training_config_file=config_path)
params.training_name = "final_model"

use_train_data = False

# Print parameters to console if needed
# params.print_params()
params.display_device_info()


########################### Configure output folder ############################

output_folder = "trainings_validation"  # Same folder for train and test preds
os.makedirs(output_folder, exist_ok=True)

# Subdirectory of output_folder where predictions are saved.
# Change this to save results for same model with different inference
# approaches.
# output_name = training_name + "_step=2"
output_name = params.training_name

save_folder = os.path.join(output_folder, output_name)
os.makedirs(save_folder, exist_ok=True)


######################## Config dataset and UNet model #########################

logger.info(f"Processing training '{params.training_name}'...")

# Define the sample IDs based on dataset size and usage
if use_train_data:
    logger.info("Predicting outputs for training data")
    if params.dataset_size == "full":
        sample_ids = [
            "01", "02", "03", "04", "06", "07", "08", "09", "11", "12", "13", "14",
            "16", "17", "18", "19", "21", "22", "23", "24", "27", "28", "29", "30",
            "33", "35", "36", "38", "39", "41", "42", "43", "44", "46"
        ]
    elif params.dataset_size == "minimal":
        sample_ids = ["01"]
else:
    logger.info("Predicting outputs for testing data")
    if params.dataset_size == "full":
        sample_ids = ["05", "10", "15", "20", "25", "32", "34", "40", "45"]
    elif params.dataset_size == "minimal":
        sample_ids = ["34"]

# Check if the specified dataset path is a directory
assert os.path.isdir(params.relative_path),\
    f'"{params.relative_path}" is not a directory'

logger.info(f"Using {params.relative_path} as dataset root path")
logger.info(f"Annotations and predictions will be saved on '{save_folder}'")

### Configure UNet ###

batch_norm = {"batch": True, "none": False}

unet_config = unet.UNetConfig(
    steps=params.unet_steps,
    first_layer_channels=params.first_layer_channels,
    num_classes=config.num_classes,
    ndims=config.ndims,
    dilation=params.dilation,
    border_mode=params.border_mode,
    batch_normalization=params.batch_normalization,
    num_input_channels=params.num_channels,
)
if not params.temporal_reduction:
    network = unet.UNetClassifier(unet_config)
else:
    assert (
        params.data_duration % params.num_channels == 0
    ), "using temporal reduction chunks_duration must be a multiple of num_channels"
    network = TempRedUNet(unet_config)

network = nn.DataParallel(network).to(params.device)

### Load UNet model ###
logger.info(
    f"Loading trained model '{params.training_name}' at epoch {params.load_epoch}...")

# Path to the saved model checkpoint
models_relative_path = "runs/"
model_path = os.path.join(models_relative_path,
                          params.training_name,
                          f"network_{params.load_epoch:06d}.pth")

# Load the model state dictionary
network.load_state_dict(torch.load(model_path,
                                   map_location=params.device))
network.eval()


############################# Run samples in UNet ##############################

for sample_id in sample_ids:
    ### Create dataset ###
    testing_dataset = SparkDataset(
        base_path=params.relative_path,
        sample_ids=[sample_id],
        testing=False,  # we just do inference, without metrics computation
        params=params,
        gt_available=True,
        inference=params.inference,
    )

    logger.info(
        f"\tTesting dataset of movie {testing_dataset.video_name} "
        f"\tcontains {len(testing_dataset)} samples."
    )

    logger.info(f"\tProcessing samples in UNet...")
    # ys and preds are numpy arrays
    _, ys, preds = get_preds(network=network,
                             test_dataset=testing_dataset,
                             compute_loss=False,
                             device=params.device)

    ### Save preds on disk ###
    logger.info(f"\tSaving annotations and predictions...")

    video_name = f"{str(params.load_epoch)}_{testing_dataset.video_name}"

    # Preds are in logarithmic scale, compute exp
    preds = np.exp(preds)

    write_videos_on_disk(
        training_name=output_name,
        video_name=video_name,
        path=save_folder,
        preds=preds,
        ys=ys,
    )

logger.info(f"DONE")
