"""
Load a saved UNet model at given epochs and save its predictions in the
folder `trainings_validation`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the
          results.

Author: Prisca Dotti
Last modified: 03.10.2023
"""

import logging
import os
import time

import torch
from torch import nn

from config import TrainingConfig, config
from data.data_processing_tools import masks_to_instances_dict, process_raw_predictions
from data.datasets import SparkDataset
from utils.in_out_tools import write_videos_on_disk

# from torch.cuda.amp import GradScaler
from utils.training_inference_tools import get_preds
from utils.training_script_utils import init_model

logger = logging.getLogger(__name__)


def main():
    config.verbosity = 3  # To get debug messages

    ####################### Get training-specific parameters #######################

    training_name = "final_model"
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
    params.training_name = training_name
    model_name = f"network_{load_epoch::06d}.pth"

    # Print parameters to console if needed
    # params.print_params()

    if testing:
        get_final_pred = True

    if get_final_pred:
        debug = True

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
    logger.info(
        f"Annotations and predictions will be saved on '{save_folder} + inference_type'"
    )

    ########################### Detect GPU, if available ###########################

    params.set_device(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    params.display_device_info()

    ######################## Config dataset and UNet model #########################

    logger.info(f"Processing training '{params.training_name}'...")

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
        logger.info("Predicting outputs for testing data")
        if params.dataset_size == "full":
            sample_ids = ["05", "10", "15", "20", "25", "32", "34", "40", "45"]
        elif params.dataset_size == "minimal":
            sample_ids = ["34"]

    # Check if the specified dataset path is a directory
    assert os.path.isdir(
        params.dataset_path
    ), f'"{params.dataset_path}" is not a directory'

    logger.info(f"Using {params.dataset_path} as dataset root path")

    ### Configure UNet ###

    network = init_model(params=params)
    network = nn.DataParallel(network).to(params.device)

    ### Load UNet model ###

    # Path to the saved model checkpoint
    models_relative_path = "runs/"
    model_path = os.path.join(models_relative_path, params.training_name, model_name)

    # Load the model state dictionary
    logger.info(f"Loading trained model '{training_name}' at epoch {load_epoch}...")
    network.load_state_dict(torch.load(model_path, map_location=params.device))
    network.eval()

    ############################# Run samples in UNet ##############################

    ############################# Run samples in UNet ##############################

    input_movies = {}
    ys = {}
    ys_instances = {}
    preds_dict = {}
    if get_final_pred:
        preds_instances = {}
        preds_segmentation = {}

    for sample_id in sample_ids:
        logger.debug(f"Processing sample {sample_id}...")
        start = time.time()
        ### Create dataset ###
        testing_dataset = SparkDataset(
            base_path=params.dataset_path,
            sample_ids=[sample_id],
            testing=testing,
            params=params,
            gt_available=True,
            inference=params.inference,
        )

        logger.info(
            f"\tTesting dataset of movie {testing_dataset.video_name} "
            f"contains {len(testing_dataset)} samples."
        )

        logger.info(f"\tProcessing samples in UNet...")
        # ys and preds are numpy arrays
        input_movies[sample_id], ys[sample_id], preds_dict[sample_id] = get_preds(
            network=network,
            test_dataset=testing_dataset,
            compute_loss=False,
            params=params,
            inference_types=inference_types,
            return_dict=True,
        )

        if testing:
            # get labelled event instances, for validation
            # ys_instances is a dict with classified event instances, for each class
            ys_instances[sample_id] = masks_to_instances_dict(
                instances_mask=testing_dataset.events,
                labels_mask=ys[sample_id],
                shift_ids=True,
            )
            # remove ignored events entry from ys_instances
            ys_instances[sample_id].pop("ignore", None)

            # get pixels labelled with 4
            # ignore_mask = np.where(ys == 4, 1, 0)

        if get_final_pred:
            ######################### get processed output #########################

            logger.debug("Getting processed output (segmentation and instances)")

            # get predicted segmentation and event instances
            if inference_types is None or len(inference_types) == 1:
                (
                    preds_instances[sample_id],
                    preds_segmentation[sample_id],
                    _,
                ) = process_raw_predictions(
                    raw_preds_dict=preds_dict[sample_id],
                    input_movie=input_movies[sample_id],
                    training_mode=False,
                    debug=debug,
                )
            else:
                # initialize empty dicts what will be indexed by inference type
                preds_instances[sample_id], preds_segmentation[sample_id] = {}, {}

                for i in inference_types:
                    logger.debug(f"\tProcessing inference type {i}...")
                    raw_preds_dict = {
                        "sparks": preds_dict[sample_id][i][1],
                        "puffs": preds_dict[sample_id][i][3],
                        "waves": preds_dict[sample_id][i][2],
                    }
                    (
                        preds_instances[sample_id][i],
                        preds_segmentation[sample_id][i],
                        _,
                    ) = process_raw_predictions(
                        preds_dict=raw_preds_dict,
                        input_movie=input_movies[sample_id],
                        training_mode=False,
                        debug=debug,
                    )

        if not get_final_pred:
            logger.info(
                f"\tTime to process sample {sample_id} in UNet: {time.time() - start:.2f} seconds."
            )
        else:
            logger.info(
                f"\tTime to process sample {sample_id} in UNet + post-processing: {time.time() - start:.2f} seconds."
            )

        ### Save preds on disk ###
        logger.info(f"\tSaving annotations and predictions...")

        video_name = f"{str(params['load_epoch'])}_{testing_dataset.video_name}"

        if inference_types is None or len(inference_types) == 1:
            write_videos_on_disk(
                training_name=output_name,
                video_name=video_name,
                path=save_folder,
                preds=[
                    None,
                    preds_dict[sample_id]["sparks"],
                    preds_dict[sample_id]["waves"],
                    preds_dict[sample_id]["puffs"],
                ],
                ys=ys[sample_id],
            )
        else:
            for i in inference_types:
                write_videos_on_disk(
                    training_name=output_name,
                    video_name=video_name,
                    path=os.path.join(save_folder, "inference_" + i),
                    preds=[
                        None,
                        preds_dict[sample_id][i]["sparks"],
                        preds_dict[sample_id][i]["waves"],
                        preds_dict[sample_id][i]["puffs"],
                    ],
                    ys=ys[sample_id],
                )

    logger.info(f"DONE")


if __name__ == "__main__":
    main()
