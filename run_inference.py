"""
Load a saved UNet model at given epochs and save its predictions in the
folder `trainings_validation`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the
          results.

Author: Prisca Dotti
Last modified: 18.10.2023
"""

import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TrainingConfig, config
from data.data_processing_tools import masks_to_instances_dict, process_raw_predictions
from utils.in_out_tools import write_videos_on_disk

# from torch.cuda.amp import GradScaler
from utils.training_inference_tools import do_inference
from utils.training_script_utils import init_dataset, init_model

logger = logging.getLogger(__name__)


def main():
    config.verbosity = 3  # To get debug messages

    ##################### Get training-specific parameters #####################

    run_name = "final_model"
    config_filename = "config_final_model.ini"
    load_epoch = 100000

    use_train_data = False
    get_final_pred = True  # set to False to only compute raw predictions
    testing = False  # set to False to only generate unet predictions
    # set to True to also compute processed outputs and metrics
    # inference_types = ['overlap', 'average', 'gaussian', 'max']
    inference_types = None  # set to None to use the default inference type from
    # the config file

    # Initialize general parameters
    params = TrainingConfig(
        training_config_file=os.path.join("config_files", config_filename)
    )
    if run_name:
        params.run_name = run_name
    model_name = f"network_{load_epoch:06d}.pth"

    # Print parameters to console if needed
    # params.print_params()

    if testing:
        get_final_pred = True

    debug = True if config.verbosity == 3 else False

    ######################### Configure output folder ##########################

    output_folder = os.path.join(
        "evaluation", "inference_script"
    )  # Same folder for train and test preds
    os.makedirs(output_folder, exist_ok=True)

    # Subdirectory of output_folder where predictions are saved.
    # Change this to save results for same model with different inference
    # approaches.
    # output_name = training_name + "_step=2"
    output_name = params.run_name

    save_folder = os.path.join(output_folder, output_name)
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Annotations and predictions will be saved on '{save_folder}'")

    ######################### Detect GPU, if available #########################

    params.set_device(device="auto")
    params.display_device_info()

    ###################### Config dataset and UNet model #######################

    logger.info(f"Processing training '{params.run_name}'...")

    # Define the sample IDs based on dataset size and usage
    if use_train_data:
        logger.info("Predicting outputs for training data")
        if params.dataset_size == "full":
            sample_ids = [
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
        elif params.dataset_size == "minimal":
            sample_ids = ["01"]
        else:
            raise ValueError(
                f"Unknown dataset size '{params.dataset_size}'. "
                f"Choose between 'full' and 'minimal'."
            )
    else:
        logger.info("Predicting outputs for testing data")
        if params.dataset_size == "full":
            sample_ids = ["05", "10", "15", "20", "25", "32", "34", "40", "45"]
        elif params.dataset_size == "minimal":
            sample_ids = ["34"]
        else:
            raise ValueError(
                f"Unknown dataset size '{params.dataset_size}'. "
                f"Choose between 'full' and 'minimal'."
            )

    # Check if the specified dataset path is a directory
    assert os.path.isdir(
        params.dataset_path
    ), f'"{params.dataset_path}" is not a directory'

    logger.info(f"Using {params.dataset_path} as dataset root path")

    # Create dataset
    dataset = init_dataset(
        params=params,
        sample_ids=sample_ids,
        inference_dataset=True,
        print_dataset_info=True,
    )

    # Create a dataloader
    dataset_loader = DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
    )

    ### Configure UNet ###

    network = init_model(params=params)
    network = nn.DataParallel(network).to(params.device)

    ### Load UNet model ###

    # Path to the saved model checkpoint
    models_relative_path = os.path.join("models", "saved_models")
    model_path = os.path.join(models_relative_path, params.run_name, model_name)

    # Load the model state dictionary
    logger.info(f"Loading trained model '{run_name}' at epoch {load_epoch}...")
    network.load_state_dict(torch.load(model_path, map_location=params.device))
    network.eval()

    ########################### Run samples in UNet ############################

    if inference_types is None:
        inference_types = [params.inference]

    # get U-Net's raw predictions
    raw_preds = do_inference(
        network=network,
        params=params,
        dataloader=dataset_loader,
        device=params.device,
        compute_loss=False,
        inference_types=inference_types,
    )

    ############# Get movies and labels (and instances if testing) #############

    xs = dataset.get_movies()
    ys = dataset.get_labels()

    if testing:
        ys_instances = dataset.get_instances()

        # convert instance masks to dictionaries
        ys_instances = {
            i: masks_to_instances_dict(
                instances_mask=instances_mask,
                labels_mask=ys[i],
                shift_ids=True,
            )
            for i, instances_mask in ys_instances.items()
        }

        # remove ignored events entry from ys_instances
        for inference in ys_instances:
            ys_instances[inference].pop("ignore", None)

    #################### Get processed output (if required) ####################

    if get_final_pred:
        logger.debug("Getting processed output (segmentation and instances)")

        final_segmentation_dict = {}
        final_instances_dict = {}
        for i in range(len(sample_ids)):
            movie_segmentation = {}
            movie_instances = {}

            for inference in inference_types:
                # transform raw predictions into a dictionary
                raw_preds_dict = {
                    event_type: raw_preds[i][inference][event_label]
                    for event_type, event_label in config.classes_dict
                }

                preds_instances, preds_segmentation, _ = process_raw_predictions(
                    raw_preds_dict=raw_preds_dict,
                    input_movie=xs[i],
                    training_mode=False,
                    debug=debug,
                )

                movie_segmentation[inference] = preds_segmentation
                movie_instances[inference] = preds_instances

            final_segmentation_dict[sample_ids[i]] = movie_segmentation
            final_instances_dict[sample_ids[i]] = movie_instances

    else:
        final_segmentation_dict = {}
        final_instances_dict = {}

    ############################ Save preds on disk ############################

    logger.info(f"\tSaving annotations and predictions...")

    for i, sample_id in enumerate(sample_ids):
        for inference in inference_types:
            video_name = f"{str(params.load_epoch)}_{sample_id}_{inference}"

            raw_preds_movie = raw_preds[i][inference]
            if get_final_pred:
                segmented_preds_movie = final_segmentation_dict[sample_id][inference]
                instances_preds_movie = final_instances_dict[sample_id][inference]
            else:
                segmented_preds_movie = None
                instances_preds_movie = None

            write_videos_on_disk(
                training_name=output_name,
                video_name=video_name,
                path=os.path.join(save_folder, "inference_" + inference),
                xs=xs[i],
                ys=ys[i],
                raw_preds=raw_preds_movie,
                segmented_preds=segmented_preds_movie,
                instances_preds=instances_preds_movie,
            )

    logger.info(f"DONE")


if __name__ == "__main__":
    main()
