""""
Script that contains a function that can be used to get 
predictions from a movie path or from a movie array in numpy 
format in Matlab.

TODO: update the code according to the meeting I had with Rado
and the updated versions of the other scripts.
"""

import os

import imageio
import napari
import numpy as np
import torch
from torch import nn

from config import TrainingConfig, config
from data.data_processing_tools import preds_dict_to_mask, process_raw_predictions
from data.datasets import SparkDatasetPath
from utils.in_out_tools import write_videos_on_disk
from utils.training_inference_tools import get_preds
from utils.training_script_utils import init_model
from utils.visualization_tools import (
    get_annotations_contour,
    get_discrete_cmap,
    get_labels_cmap,
)


def main():
    ### Set training-specific parameters ###

    # Initialize training-specific parameters
    config_path = os.path.join("config_files", "config_final_model.ini")
    params = TrainingConfig(training_config_file=config_path)
    params.training_name = "final_model"
    model_name = f"network_100000.pth"

    assert params.nn_architecture in [
        "pablos_unet",
        "github_unet",
        "openai_unet",
    ], f"nn_architecture must be one of 'pablos_unet', 'github_unet', 'openai_unet'"

    ### Configure UNet ###
    params.set_device(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    network = init_model(params=params)
    network = nn.DataParallel(network).to(params.device)

    ### Load UNet model ###

    # Path to the saved model checkpoint
    models_relative_path = "runs/"
    model_path = os.path.join(models_relative_path, params.training_name, model_name)

    # Load the model state dictionary
    network.load_state_dict(torch.load(model_path, map_location=params.device))
    network.eval()

    # Define movie path
    movie_path = os.path.join(
        r"C:\Users\dotti\sparks_project\data\sparks_dataset", "05_video.tif"
    )

    # Function definition
    @torch.no_grad()
    def get_preds_from_path(
        model, params, movie_path, return_dict=False, output_dir=None
    ):
        """
        Function to get predictions from a movie path.

        Args:
        - model: Model to use for prediction.
        - params: Training parameters for prediction.
        - movie_path: Path to the movie.
        - return_dict: If True, return a dictionary; else return a tuple of numpy arrays.
        - output_dir: If not None, save raw predictions on disk.

        Returns:
        - If return_dict is True, return a dictionary with keys 'sparks', 'puffs', 'waves';
        else return a tuple of numpy arrays with integral values for classes and instances.
        """

        ### Get sample as dataset ###
        sample_dataset = SparkDatasetPath(
            sample_path=movie_path,
            params=params,
            ignore_index=config.ignore_index,
            # resampling=False, # It could be implemented later
            # resampling_rate=150,
        )

        ### Run sample in UNet ###
        input_movie, preds_dict = get_preds(
            network=model,
            test_dataset=sample_dataset,
            compute_loss=False,
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

        ## Save raw preds on disk ### I don't know if this is necessary
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

    segmentation, instances = get_preds_from_path(
        model=network,
        params=params,
        movie_path=movie_path,
        return_dict=False,
    )

    ### Visualize preds with Napari

    # open original movie
    sample = np.asarray(imageio.volread(movie_path))

    # set up napari parameters
    cmap = get_discrete_cmap(name="gray", lut=16)
    labels_cmap = get_labels_cmap()

    # visualize only border of classes (segmentation array)
    segmentation_border = get_annotations_contour(segmentation)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(sample, name="input movie", colormap=("colors", cmap))

        viewer.add_labels(
            segmentation_border,
            name="segmentation",
            opacity=0.9,
            color=labels_cmap,
        )  # only visualize border

        # viewer.add_labels(
        #   segmentation_border,
        #   name='segmentation'
        #   opacity=0.5,
        #   color=labels_cmap,
        # ) # to visualize whole roi instead

        viewer.add_labels(
            instances,
            name="instances",
            opacity=0.5,
        )


if __name__ == "__main__":
    main()
