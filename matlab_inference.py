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
from utils.training_inference_tools import get_preds, get_preds_from_path
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
        r"C:\Users\prisc\Code\sparks_project\data\sparks_dataset\34_video.tif"
        # r"C:\Users\dotti\sparks_project\data\sparks_dataset", "05_video.tif"
    )

    ### Get predictions from movie path ###

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
