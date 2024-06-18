"""
**Author**: Prisca Dotti# 
**Last Edit**: 17.06.2024

UPDATES:
- 17.06.2024: Adapted the script to the PatchSparksDataset class.

The dataset used for training could be one of the following:
- dataset of patches extracted in a meaningful way from confocal imaging
  recordings
- dataset of recordings from which patches are extracted when processing them in
  the U-Net for memory management reasons and which are recombined as whole
  movies after inference
- dataset of simulated confocal imaging patches --> this is similar to what I've
  been doing so far
- a combination of real and fake patches of confocal imaging data

Usage:
    python -m patches_experiments.run_training_patches config\config_patches.ini
    from the root directory of the repository.

"""

import logging
import os
import random

import numpy as np
import torch
from torch import nn, optim

# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from config import TrainingConfig, config
from data.datasets import PatchCaEventsDataset, PatchSparksDataset
from models.UNet import unet
from utils.training_inference_tools import (
    MyTrainingManager,
    TransformedSparkDataset,
    random_flip,
    random_flip_noise,
    sampler,
    test_function,
    training_step,
    weights_init,
)
from utils.training_script_utils import (
    get_sample_ids,
    init_config_file_path,
    init_criterion,
    init_dataset,
    init_model,
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

##################### Get training-specific parameters #####################

# Initialize training-specific parameters
# (get the configuration file path from ArgParse)
params = TrainingConfig(training_config_file=init_config_file_path())

# Print parameters to console if needed
params.print_params()

######################### Initialize random seeds ##########################

# We used these random seeds to ensure reproducibility of the results

torch.manual_seed(0)  # <--------------------------------------------------!
random.seed(0)  # <--------------------------------------------------------!
np.random.seed(0)  # <-----------------------------------------------------!

############################ Configure datasets ############################

# Select samples for training and testing based on dataset size
train_sample_ids = get_sample_ids(
    train_data=True,
    dataset_size=params.dataset_size,
)
test_sample_ids = get_sample_ids(
    train_data=False,
    dataset_size=params.dataset_size,
)

# Initialize training dataset: here it only samples random patches
# dataset = PatchCaEventsDataset(
dataset = PatchSparksDataset(
    params=params,
    base_path=params.dataset_dir,
    sample_ids=train_sample_ids,
    load_instances=False,
    inference=None,
)
# Apply transforms based on noise_data_augmentation setting
# (transforms are applied when getting a sample from the dataset)
transforms = random_flip_noise if params.noise_data_augmentation else random_flip
dataset = TransformedSparkDataset(dataset, transforms)

logger.info(f"Samples in dataset (patches sampled in an iteration): {len(dataset)}")

# Initialize testing datasets
testing_dataset = init_dataset(
    params=params,
    sample_ids=test_sample_ids,
    apply_data_augmentation=False,
    load_instances=True,
)
# Initialize data loaders
dataset_loader = DataLoader(
    dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers,
    pin_memory=params.pin_memory,
)

############################## Configure UNet ##############################

# Initialize the UNet model
network = init_model(params=params)

# Move the model to the GPU if available
if params.device.type != "cpu":
    network = nn.DataParallel(network).to(params.device, non_blocking=True)
    # cudnn.benchmark = True

# Watch the model with wandb for logging if enabled
if params.wandb_log:
    wandb.watch(network)

# Initialize UNet weights if required
if params.initialize_weights:
    logger.info("Initializing UNet weights...")
    network.apply(weights_init)

# The following line is commented as it does not work on Windows
# torch.compile(network, mode="default", backend="inductor")

########################### Initialize training ############################

# Initialize the optimizer based on the specified type
if params.optimizer == "adam":
    optimizer = optim.Adam(network.parameters(), lr=params.lr_start)
elif params.optimizer == "adadelta":
    optimizer = optim.Adadelta(network.parameters(), lr=params.lr_start)
elif params.optimizer == "sgd":
    optimizer = optim.SGD(network.parameters(), lr=params.lr_start)
else:
    logger.error(f"{params.optimizer} is not a valid optimizer.")
    exit()

# Initialize the learning rate scheduler if specified
if params.scheduler == "step":
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.scheduler_step_size,
        gamma=params.scheduler_gamma,
    )
else:
    scheduler = None

# Define the output directory path
output_path = os.path.join(config.output_dir, params.run_name)
logger.info(f"Output directory: {os.path.realpath(output_path)}")

# Initialize the summary writer for TensorBoard logging
summary_writer = SummaryWriter(os.path.join(output_path, "summary"), purge_step=0)

# Check if a pre-trained model should be loaded
if params.load_run_name != "":
    load_path = os.path.join(config.output_dir, params.load_run_name)
    logger.info(f"Model loaded from directory: {os.path.realpath(load_path)}")
else:
    load_path = None

# Initialize the loss function
criterion = init_criterion(params=params, dataset=dataset)

# Create a directory to save predicted class movies
preds_output_dir = os.path.join(output_path, "predictions")
os.makedirs(preds_output_dir, exist_ok=True)

# Create a dictionary of managed objects
managed_objects = {"network": network, "optimizer": optimizer}
if scheduler is not None:
    managed_objects["scheduler"] = scheduler

# Create a training manager with the specified training and testing functions
trainer = MyTrainingManager(
    # Training parameters
    training_step=lambda _: training_step(
        dataset_loader=dataset_loader,
        params=params,
        sampler=sampler,
        network=network,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        # scaler=GradScaler(),
    ),
    save_every=params.c.getint("training", "save_every", fallback=5000),
    load_path=load_path,
    save_path=output_path,
    managed_objects=unet.managed_objects(managed_objects),
    # Testing parameters
    test_function=lambda _: test_function(
        network=network,
        device=params.device,
        criterion=criterion,
        params=params,
        testing_dataset=testing_dataset,
        training_name=params.run_name,
        output_dir=preds_output_dir,
        training_mode=True,
    ),
    test_every=params.c.getint("training", "test_every", fallback=1000),
    plot_every=params.c.getint("training", "test_every", fallback=1000),
    summary_writer=summary_writer,
)

# Load the model if a specific epoch is provided
if params.load_epoch != 0:
    trainer.load(params.load_epoch)

############################## Start training ##############################

# Set the network in training mode
network.train()

# Resume the W&B run if needed (commented out for now)
# if wandb.run.resumed:
#     checkpoint = torch.load(wandb.restore(checkpoint_path))
#     network.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']

# Check if training is enabled in the configuration
if params.c.getboolean("general", "training", fallback=False):
    # Validate the network before training if resuming from a checkpoint
    if params.load_epoch > 0:
        logger.info("Validate network before training")
        trainer.run_validation(wandb_log=params.wandb_log)

    logger.info("Starting training")
    # Train the model for the specified number of epochs
    trainer.train(
        params.train_epochs,
        print_every=params.c.getint("training", "print_every", fallback=100),
        wandb_log=params.wandb_log,
    )

# Check if final testing/validation is enabled in the configuration
if params.c.getboolean("general", "testing", fallback=False):
    logger.info("Starting final validation")
    # Run the final validation/testing procedure
    trainer.run_validation(wandb_log=params.wandb_log)
