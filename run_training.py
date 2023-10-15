"""
Run this script to train the UNet model on the given dataset.
Parameters specific to the project can be set in the config.py file.
Parameters specific to the training procedure can be set in the specified
config.ini file.

To set the config file from ArgParse. I.e., run the script from the
command line such as:
python training.py  /path/to/config_file.ini

Author: Prisca Dotti
Last modified: 28.09.2023
"""

import logging
import os

import torch
import wandb
from torch import nn, optim

# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainingConfig, config
from models.UNet import unet
from utils.training_inference_tools import (
    MyTrainingManager,
    sampler,
    test_function,
    training_step,
    weights_init,
)
from utils.training_script_utils import (
    init_config_file_path,
    init_criterion,
    init_dataset,
    init_model,
    init_testing_dataset,
)

logger = logging.getLogger(__name__)


def main():
    torch.set_float32_matmul_precision("high")

    ##################### Get training-specific parameters #####################

    # Initialize training-specific parameters
    # (get the configuration file path from ArgParse)
    params = TrainingConfig(training_config_file=init_config_file_path())

    # Print parameters to console if needed
    params.print_params()

    ######################### Initialize random seeds ##########################

    # We used these random seeds to ensure reproducibility of the results

    # torch.manual_seed(0) <--------------------------------------------------!
    # random.seed(0) <--------------------------------------------------------!
    # np.random.seed(0) <-----------------------------------------------------!

    ############################ Configure datasets ############################

    # TODO: magari devo creare un'altra classe nel file config dove specifico
    # come dev'essere strutturato il dataset (e.g., che nome devono avere i files,
    # dove sono salvati, come splittare il test/val/train dataset, etc.)

    # Select samples for training and testing based on dataset size
    if params.dataset_size == "full":
        train_sample_ids = [
            "01",
            "02",
            "03",
            "04",
            "06",
            "07",
            "08",
            "09",
            "11",
            "12",
            "13",
            "14",
            "16",
            "17",
            "18",
            "19",
            "21",
            "22",
            "23",
            "24",
            "27",
            "28",
            "29",
            "30",
            "33",
            "35",
            "36",
            "38",
            "39",
            "41",
            "42",
            "43",
            "44",
            "46",
        ]
        test_sample_ids = ["05", "10", "15", "20", "25", "32", "34", "40", "45"]
    elif params.dataset_size == "minimal":
        train_sample_ids = ["01"]
        test_sample_ids = ["34"]
    else:
        logger.error(f"{params.dataset_size} is not a valid dataset size.")
        exit()

    # Initialize training dataset
    dataset = init_dataset(
        params=params,
        sample_ids=train_sample_ids,
    )

    # Initialize testing datasets
    testing_datasets = init_testing_dataset(
        params=params,
        test_sample_ids=test_sample_ids,
    )

    # Initialize data loaders
    dataset_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
    )

    ############################## Configure UNet ##############################

    # Initialize the UNet model
    network = init_model(params=params)

    # Move the model to the GPU if available
    if params.device != "cpu":
        network = nn.DataParallel(network).to(params.device, non_blocking=True)
        torch.backends.cudnn.benchmark = True

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

    # Set the network in training mode
    network.train()

    # Define the output directory path
    output_path = os.path.join(config.output_relative_path, params.run_name)
    logger.info(f"Output directory: {output_path}")

    # Initialize the summary writer for TensorBoard logging
    summary_writer = SummaryWriter(os.path.join(output_path, "summary"), purge_step=0)

    # Check if a pre-trained model should be loaded
    if params.load_run_name != None:
        load_path = os.path.join(config.output_relative_path, params.load_run_name)
        logger.info(f"Model loaded from directory: {load_path}")
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
            sampler=sampler,
            network=network,
            optimizer=optimizer,
            # scaler=GradScaler(),
            scheduler=scheduler,
            criterion=criterion,
            dataset_loader=dataset_loader,
            params=params,
        ),
        save_every=params.c.getint("training", "save_every", fallback=5000),
        load_path=load_path,
        save_path=output_path,
        managed_objects=unet.managed_objects(managed_objects),
        # Testing parameters
        test_function=lambda _: test_function(
            network=network,
            criterion=criterion,
            testing_datasets=testing_datasets,
            params=params,
            training_name=params.run_name,
            output_dir=preds_output_dir,
            training_mode=True,
            debug=config.debug_mode,
        ),
        test_every=params.c.getint("training", "test_every", fallback=1000),
        plot_every=params.c.getint("training", "test_every", fallback=1000),
        summary_writer=summary_writer,
    )

    ############################## Start training ##############################

    # Load the model if a specific epoch is provided
    if params.load_epoch != 0:
        trainer.load(params.load_epoch)

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


# # Shutdown computer after training
# import subprocess
# subprocess.run(["shutdown", "-s"])

if __name__ == "__main__":
    main()
