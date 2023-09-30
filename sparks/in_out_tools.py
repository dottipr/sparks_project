"""
Script with functions to either load data or save data to disc.

Author: Prisca Dotti
Last modified: 28.09.2023
"""

import csv
import datetime
import logging
import os

import imageio
import numpy as np
from data_processing_tools import process_spark_prediction
from metrics_tools import correspondences_precision_recall
from visualization_tools import (
    add_colored_classes_to_video,
    add_colored_instances_to_video,
    add_colored_paired_sparks_to_video,
)

from config import config

# REMARK
# How to save .json files:
# with open(filename,"w") as f:
#    json.dump(dict_or_data,f)

logger = logging.getLogger(__name__)


################################ Loading utils #################################


def load_movies_ids(data_folder, ids, names_available=False, movie_names=None):
    """
    Load movies corresponding to a given list of indices.

    Args:
        data_folder (str): Folder where movies are saved, movies are saved as
            "[0-9][0-9]*.tif".
        ids (list): List of movie IDs (of the form "[0-9][0-9]").
        names_available (bool): If True, can specify the name of the movie file,
            such as "XX_<movie_name>.tif".
        movie_names (str, optional): Movie name, if available.

    Returns:
        dict: A dictionary containing loaded movies indexed by video ID.
    """
    xs_all_trainings = {}

    if names_available:
        xs_filenames = [os.path.join(data_folder, f"{idx}_{movie_names}.tif")
                        for idx in ids]
    else:
        xs_filenames = [
            os.path.join(data_folder, movie_name)
            for movie_name in os.listdir(data_folder)
            if movie_name.startswith(tuple(ids))
        ]

    for f in xs_filenames:
        video_id = os.path.split(f)[1][:2]
        xs_all_trainings[video_id] = np.asarray(imageio.volread(f))

    return xs_all_trainings


def load_annotations_ids(data_folder, ids, mask_names="video_mask"):
    """
    Load annotations for a list of movie IDs.

    Args:
        data_folder (str): The folder where annotations are saved.
        ids (list): List of movie IDs to be considered.
        mask_names (str, optional): Name of the type of masks to load.

    Returns:
        dict: A dictionary containing loaded annotations indexed by movie IDs.
    """
    ys_all_trainings = {}

    ys_filenames = [
        os.path.join(data_folder, f"{idx}_{mask_names}.tif") for idx in ids
    ]

    for f in ys_filenames:
        video_id = os.path.split(f)[1][:2]
        ys_all_trainings[video_id] = np.asarray(imageio.volread(f), dtype=int)

    # check that annotations aren't empty
    assert len(ys_all_trainings) > 0, "No annotations found."

    return ys_all_trainings


def load_rgb_annotations_ids(data_folder, ids, mask_names="separated_events"):
    """
    Load RGB annotations with separated events for a list of movie IDs.

    Args:
        data_folder (str): The folder where RGB annotations are saved.
        ids (list): List of movie IDs to be considered.
        mask_names (str, optional): Name of the type of RGB masks to load.

    Returns:
        dict: A dictionary containing loaded RGB annotations indexed by movie IDs.
    """
    ys_all_trainings = {}

    ys_filenames = [
        os.path.join(data_folder, f"{idx}_{mask_names}.tif") for idx in ids
    ]

    # Integer representing white colour in RGB mask
    white_int = 255 * 255 * 255 + 255 * 255 + 255

    for f in ys_filenames:
        video_id = os.path.split(f)[1][:2]
        rgb_video = np.asarray(imageio.volread(f), dtype=int)

        mask_video = (
            255 * 255 * rgb_video[..., 0] + 255 *
            rgb_video[..., 1] + rgb_video[..., 2]
        )

        mask_video[mask_video == white_int] = 0

        ys_all_trainings[video_id] = mask_video

    return ys_all_trainings


# OLD
# def load_predictions_ids(training_name, epoch, metrics_folder, ids):
#     """
#     Load processed annotations, predicted sparks, puffs, and waves for a given
#     training.

#     Args:
#         training_name (str): Saved training name.
#         epoch (int): Training epoch to load predictions for.
#         metrics_folder (str): Folder where predictions and annotations are saved.
#         ids (list): List of movie IDs to be considered.

#     Returns:
#         Tuple: A tuple containing dictionaries of loaded data for each movie:
#                (training_ys, training_sparks, training_puffs, training_waves).
#     """
#     # Import .tif files as numpy array
#     base_name = os.path.join(metrics_folder, f"{training_name}_{epoch}_")

#     if "temporal_reduction" in training_name:
#         logger.warning("Method is using temporal reduction, "
#                        "processed annotations have a different shape.")

#     # Get predictions and annotations filenames
#     ys_filenames = sorted([f"{base_name}{sample_id}_ys.tif"
#                            for sample_id in ids])
#     sparks_filenames = sorted([f"{base_name}{sample_id}_sparks.tif"
#                                for sample_id in ids])
#     puffs_filenames = sorted([f"{base_name}{sample_id}_puffs.tif"
#                               for sample_id in ids])
#     waves_filenames = sorted([f"{base_name}{sample_id}_waves.tif"
#                               for sample_id in ids])

#     # Create dictionaires to store loaded data for each movie
#     training_ys = {}
#     training_sparks = {}
#     training_puffs = {}
#     training_waves = {}

#     for y, s, p, w in zip(
#         ys_filenames, sparks_filenames, puffs_filenames, waves_filenames
#     ):
#         video_id = os.path.split(y)[1][:2]

#         ys_loaded = np.asarray(imageio.volread(y), dtype=int)
#         training_ys[video_id] = ys_loaded

#         if "temporal_reduction" in training_name:
#             logger.info(
#                 "Training using temporal reduction, extending predictions...")
#             s_preds = np.asarray(imageio.volread(s))
#             p_preds = np.asarray(imageio.volread(p))
#             w_preds = np.asarray(imageio.volread(w))

#             # Repeat predicted frames 4 times
#             s_preds = np.repeat(s_preds, 4, 0)
#             p_preds = np.repeat(p_preds, 4, 0)
#             w_preds = np.repeat(w_preds, 4, 0)

#             training_sparks[video_id] = s_preds
#             training_puffs[video_id] = p_preds
#             training_waves[video_id] = w_preds
#         else:
#             training_sparks[video_id] = np.asarray(imageio.volread(s))
#             training_puffs[video_id] = np.asarray(imageio.volread(p))
#             training_waves[video_id] = np.asarray(imageio.volread(w))

#     return training_ys, training_sparks, training_puffs, training_waves

# OLD
# def load_predictions_all_trainings_ids(training_names, epochs, metrics_folder, ids):
#     """
#     Load processed annotations, predicted sparks, puffs, and waves for a list of
#     training names.

#     Args:
#         training_names (list): List of saved training names.
#         epochs (list): List of training epochs to load predictions for
#         (corresponding to the training names).
#         metrics_folder (str): Folder where predictions and annotations are saved.
#         ids (list): List of movie IDs to be considered.

#     Returns:
#         Tuple: A tuple containing dictionaries of loaded data for each movie:
#                (ys, sparks, puffs, waves).
#     """
#     ys = {}
#     s = {}  # sparks
#     p = {}  # puffs
#     w = {}  # waves

#     for name, epoch in zip(training_names, epochs):
#         ys[name], s[name], p[name], w[name] = load_predictions_ids(
#             name, epoch, metrics_folder, ids
#         )

#     return ys, s, p, w


####################### Tools for writing videos on disc #######################


def write_videos_on_disk(
    training_name, video_name, path="predictions", xs=None, ys=None, preds=None
):
    """
    Write videos to disk.

    Args:
        training_name (str): Training name.
        video_name (str): Video name.
        path (str): Output directory path.
        xs (numpy.ndarray, optional): Input video used by the network.
        ys (numpy.ndarray, optional): Segmentation video used in the loss function.
        preds (list of numpy.ndarray, optional): All U-Net predictions 
            [bg preds, sparks preds, puffs preds, waves preds] (should already be
            normalized between 0 and 1).

    Returns:
        None
    """
    if training_name is not None:
        out_name_root = f"{training_name}_{video_name}_"
    else:
        out_name_root = f"{video_name}_"

    logger.debug(f"Writing videos on directory {os.path.abspath(path)} ..")
    os.makedirs(os.path.abspath(path), exist_ok=True)

    if xs is not None:
        imageio.volwrite(os.path.join(
            path, out_name_root + "xs.tif"), xs)
    if ys is not None:
        imageio.volwrite(os.path.join(
            path, out_name_root + "ys.tif"), np.uint8(ys))
    if preds is not None:
        imageio.volwrite(os.path.join(
            path, out_name_root + "sparks.tif"), preds[1])
        imageio.volwrite(os.path.join(
            path, out_name_root + "waves.tif"), preds[2])
        imageio.volwrite(os.path.join(
            path, out_name_root + "puffs.tif"), preds[3])


def write_colored_events_videos_on_disk(
    movie,
    events_mask,
    out_dir,
    movie_fn,
    transparency=50,
    ignore_frames=0,
    white_bg=False,
    instances=False,
    label_mask=None
):
    """
    Paste colored segmentation on a video and save it on disk.
    Color used:
    sparks (1): green
    puffs (3): red
    waves (2): purple
    ignore regions (4): grey

    Args:
        movie (numpy.ndarray): Input video.
        events_mask (numpy.ndarray): Class segmentation (labels or preds).
        out_dir (str): Output directory.
        movie_fn (str): Video filename.
        transparency (int, optional): Transparency level of the colored segmentation.
        ignore_frames (int, optional): Number of frames to ignore.
        white_bg (bool, optional): Save colored segmentation on a white background.
        instances (bool, optional): True if events_mask is a mask of instances,
            False for classes.
        label_mask (numpy.ndarray, optional): If provided, use label mask for
            contour drawing.

    Returns:
        None
    """
    if instances:
        colored_movie = add_colored_instances_to_video(movie=movie,
                                                       instances_mask=events_mask,
                                                       transparency=transparency,
                                                       ignore_frames=ignore_frames,
                                                       white_bg=white_bg)
    else:
        colored_movie = add_colored_classes_to_video(movie=movie,
                                                     classes_mask=events_mask,
                                                     transparency=transparency,
                                                     ignore_frames=ignore_frames,
                                                     white_bg=white_bg,
                                                     label_mask=label_mask)

    imageio.volwrite(os.path.join(out_dir, movie_fn + ".tif"), colored_movie)


############################### csv files tools ################################

# OLD
# def create_csv(filename, positions):
#     """
#     Create a CSV file with frame, x, and y columns from a 3D positions array.

#     Args:
#         filename (str): The name of the CSV file to be created.
#         positions (numpy.ndarray): A 3D array containing position data.

#     Returns:
#         None
#     """
#     with open(filename, "w", newline="") as csvfile:
#         filewriter = csv.writer(
#             csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
#         filewriter.writerow(["frame", "x", "y"])

#         for loc in positions:
#             frame, x, y = loc[0], loc[2], loc[1]
#             filewriter.writerow([frame, x, y])
#             logger.info(f"Location [{frame}, {x}, {y}] written to .csv file")


################## tools for writing sparks locations on disk ##################


# OLD
# def write_spark_locations_on_disk(
#     spark_pred, movie, filename, t_detection_sparks, min_r_sparks, ignore_frames=0
# ):
#     """
#     Write detected spark locations to a CSV file.

#     Args:
#         spark_pred (np.ndarray): Predicted spark locations.
#         movie (np.ndarray): Input video used for prediction.
#         filename (str): Path to the output CSV file.
#         t_detection_sparks (float): Threshold for spark detection.
#         min_r_sparks (int): Minimum radius for spark detection.
#         ignore_frames (int, optional): Number of frames to ignore at the beginning.

#     Returns:
#         None
#     """
#     # Process predictions
#     sparks_list = process_spark_prediction(
#         pred=spark_pred,
#         movie=movie,
#         ignore_frames=ignore_frames,
#         t_detection=t_detection_sparks,
#         min_radius=min_r_sparks,
#     )

#     logger.info(f"Writing sparks locations to .csv file in file {filename}")
#     create_csv(filename, sparks_list)

# OLD
# def write_paired_sparks_on_disk(paired_true, paired_preds, fp, fn, file_path):
#     """
#     Write spark peak correspondences and classification results to a text file.

#     Args:
#         paired_true (list): List of coordinates of annotated sparks paired with
#             predictions.
#         paired_preds (list): List of coordinates of predicted sparks paired with
#             annotations.
#         fp (list): List of coordinates of false positive predicted sparks.
#         fn (list): List of coordinates of false negative annotated sparks.
#         file_path (str): Location and filename of the output text file.

#     Returns:
#         None
#     """
#     with open(file_path, "w") as f:
#         f.write(f"{datetime.datetime.now()}\n\n")
#         f.write(f"Paired annotations and preds:\n")
#         for p_true, p_preds in zip(paired_true, paired_preds):
#             f.write(f"{[int(coord) for coord in p_true]} {[int(coord) for coord in p_preds]}\n")
#         f.write(f"\n")
#         f.write(f"Unpaired preds (false positives):\n")
#         for f_p in fp:
#             f.write(f"{[int(coord) for coord in f_p]}\n")
#         f.write(f"\n")
#         f.write(f"Unpaired annotations (false negatives):\n")
#         for f_n in fn:
#             f.write(f"{[int(coord) for coord in f_n]}\n")

# OLD
# def write_colored_sparks_on_disk(
#     training_name,
#     video_name,
#     paired_real,
#     paired_pred,
#     false_positives,
#     false_negatives,
#     path="predictions",
#     xs=None,
#     movie_shape=None,
# ):
#     """
#     Write input video with colored paired sparks and a text file with sparks
#     coordinates on disk.

#     Args:
#         training_name (str): The name of the training.
#         video_name (str): The name of the video.
#         paired_real (list): List of coordinates [t, y, x] of paired annotated sparks.
#         paired_pred (list): List of coordinates [t, y, x] of paired predicted sparks.
#         false_positives (list): List of coordinates [t, y, x] of wrongly predicted
#             sparks.
#         false_negatives (list): List of coordinates [t, y, x] of not found annotated
#             sparks.
#         path (str, optional): Directory where the output will be saved.
#         xs (np.ndarray, optional): Input video used by the network. If None, sparks
#             will be saved on a white background.
#         movie_shape (tuple, optional): If input movie xs is None, provide the video
#             shape (t, y, x).

#     Returns:
#         None
#     """
#     if xs is not None:
#         sample_video = xs
#     else:
#         assert movie_shape is not None,\
#             "Provide movie shape if not providing input movie."
#         sample_video = 255 * np.ones(movie_shape)

#     # Compute colored sparks mask
#     transparency = 45
#     annotated_video = add_colored_paired_sparks_to_video(
#         movie=sample_video,
#         paired_true=paired_real,
#         paired_preds=paired_pred,
#         fp=false_positives,
#         fn=false_negatives,
#         transparency=transparency,
#     )

#     # Set saved movies filenames
#     white_background_fn = "white_BG" if xs is None else ""
#     out_name_root = f"{training_name }_{video_name}_{white_background_fn}"

#     # Save video on disk
#     imageio.volwrite(
#         os.path.join(
#             path, f"{out_name_root}_colored_sparks.tif"), annotated_video
#     )

#     # Write sparks locations to file
#     file_path = os.path.join(path, f"{out_name_root}_sparks_location.txt")
#     write_paired_sparks_on_disk(
#         paired_true=paired_real,
#         paired_preds=paired_pred,
#         fp=false_positives,
#         fn=false_negatives,
#         file_path=file_path,
#     )

# OLD
# def pair_and_write_sparks_on_video(
#     movie,
#     coords_true,
#     coords_preds,
#     out_path,
#     white_background,
#     transparency=50,
#     training_name=None,
#     epoch=None,
#     movie_id=None,
# ):
#     """
#     Compute annotated and predicted spark peaks correspondences and save the
#     resulting paired coordinates, false negatives, and false positives on a text
#     file. Additionally, save colored spark peaks on a movie on disk.

#     Args:
#         movie (np.ndarray): Input movie.
#         coords_true (list): List of coordinates [t, y, x] of annotated sparks.
#         coords_preds (list): List of coordinates [t, y, x] of predicted sparks.
#         out_path (str): Directory where the output will be saved.
#         white_background (bool): Whether to use a white background for the movie.
#         transparency (int, optional): Transparency of colored sparks.
#         training_name (str, optional): Name of the training.
#         epoch (int, optional): Training epoch.
#         movie_id (str, optional): Movie ID.

#     Returns:
#         None
#     """
#     # Compute correspondences between annotations and predictions
#     paired_true, paired_preds, fp, fn = correspondences_precision_recall(
#         coords_true, coords_preds, return_pairs_coords=True
#     )

#     # Add colored annotations to video
#     sample_video = np.copy(movie)
#     if white_background:
#         sample_video.fill(255)  # The movie will be white

#     annotated_video = add_colored_paired_sparks_to_video(
#         movie=sample_video,
#         paired_true=paired_true,
#         paired_preds=paired_preds,
#         fp=fp,
#         fn=fn,
#         transparency=transparency,
#     )

#     # Write sparks locations to file
#     coords_file_path = os.path.join(
#         out_path, f"{movie_id}_sparks_location.txt")
#     write_paired_sparks_on_disk(
#         paired_true=paired_true,
#         paired_preds=paired_preds,
#         fp=fp,
#         fn=fn,
#         file_path=coords_file_path,
#     )

#     # Set saved movies filenames
#     wb_fn = "_white_backgroud" if white_background else ""
#     movie_fn = f"colored_sparks{wb_fn}.tif"

#     if movie_id is not None:
#         movie_fn = f"{movie_id}_{movie_fn}"
#     if epoch is not None:
#         movie_fn = f"{epoch}_{movie_fn}"
#     if training_name is not None:
#         movie_fn = f"{training_name}_{movie_fn}"

#     movie_path = os.path.join(out_path, movie_fn)
#     imageio.volwrite(movie_path, annotated_video)

#     # Write summary file with parameters
#     summary_file_path = os.path.join(out_path, f"parameters{wb_fn}.txt")

#     with open(summary_file_path, "w") as f:
#         f.write(f"{datetime.datetime.now()}\n\n")

#         f.write("Phyisiological parameters\n")
#         f.write(f"Pixel size: {config.pixel_size} um\n")
#         f.write(f"Min distance (x,y): {config.min_dist_xy} pixels\n")
#         f.write(f"Time frame: {config.time_frame} ms\n")
#         f.write(f"Min distance t: {config.min_dist_t} pixels\n\n")

#         if training_name is not None:
#             f.write("Training parameters\n")
#             f.write(f"Training name: {training_name}\n")
#             f.write(f"Loaded epoch: {epoch}\n")

#         f.write("Coloured sparks parameters\n")
#         f.write(f"Saved coloured sparks path: {out_path}\n")
#         f.write(f"Coloured sparks' transparency: {transparency}\n")
#         f.write(
#             f"Using white background instead of original movies: {white_background}\n"
#         )
