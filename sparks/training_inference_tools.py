"""
Functions that are used during the training of the neural network
"""

import logging
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from data_processing_tools import (
    class_to_nb,
    empty_marginal_frames,
    empty_marginal_frames_from_coords,
    get_argmax_segmented_output,
    get_event_instances_class,
    get_processed_result,
    simple_nonmaxima_suppression,
    sparks_connectivity_mask,
)
from in_out_tools import write_colored_sparks_on_disk, write_videos_on_disk
from metrics_tools import (
    compute_f_score,
    compute_iou,
    compute_puff_wave_metrics,
    correspondences_precision_recall,
    get_confusion_matrix,
    get_score_matrix,
)
from scipy import ndimage as ndi
from torch import nn

logger = logging.getLogger(__name__)


################################ training step #################################

# Make one step of the training (update parameters and compute loss)
def training_step(
    sampler,
    network,
    optimizer,
    device,
    criterion,
    dataset_loader,
    ignore_frames,
    wandb_log,
):
    # start = time.time()

    network.train()

    x, y = sampler(dataset_loader)
    x = x.to(device)  # [1, 256, 64, 512]
    y = y.to(device)  # [1, 256, 64, 512]

    # detect nan in tensors
    # if (torch.isnan(x).any() or torch.isnan(y).any()):
    #    logger.info(f"Detect nan in network input: {torch.isnan(x).any()}")
    #    logger.info(f"Detect nan in network annotation: {torch.isnan(y).any()}")

    y_pred = network(x[:, None])  # [1, 4, 256, 64, 512]

    # Compute loss
    loss = criterion(
        y_pred[..., ignore_frames:-ignore_frames],
        y[..., ignore_frames:-ignore_frames].long(),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if wandb_log:
        wandb.log({"U-Net training loss": loss.item()})

    # end = time.time()
    # print(f"Runtime for 1 training step: {end-start}")

    return {"loss": loss.item()}


# Iterator (?) over the dataset
def mycycle(dataset_loader):
    while True:
        for x in dataset_loader:
            yield x


# _cycle = mycycle(dataset_loader)


def sampler(dataset_loader):
    return next(mycycle(dataset_loader))  # (_cycle)


### functions for data augmentation ###


def random_flip(x, y):
    # flip movie and annotation mask
    # if np.random.uniform() > 0.5:
    rand = torch.rand(1).item()

    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)
        y = y.flip(-1)

    if torch.rand(1).item() > 0.5:
        x = x.flip(-2)
        y = y.flip(-2)

    return x, y


def random_flip_noise(x, y):
    # flip movie and annotation mask
    x, y = random_flip(x, y)

    # add noise to movie with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of being normal or Poisson noise
        if torch.rand(1).item() > 0.5:
            noise = torch.normal(mean=0.0, std=1.0, size=x.shape)
            x = x + noise
        else:
            x = torch.poisson(x)  # non so se funziona !!!

        # x = x.astype('float32')

    # denoise input with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of gaussian filtering or median filtering
        if torch.rand(1).item() > 0.5:
            x = ndi.gaussian_filter(x, sigma=1)
        else:
            x = ndi.median_filter(x, size=2)

    return torch.tensor(x), torch.tensor(y)


### functions related to UNet and loss weights ###


def compute_class_weights(dataset, w0=1, w1=1, w2=1, w3=1):
    # For 4 classes
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with torch.no_grad():
        for _, y in dataset:
            count0 += torch.count_nonzero(y == 0)
            count1 += torch.count_nonzero(y == 1)
            count2 += torch.count_nonzero(y == 2)
            count3 += torch.count_nonzero(y == 3)

    total = count0 + count1 + count2 + count3

    w0_new = w0 * total / (4 * count0) if count0 != 0 else 0
    w1_new = w1 * total / (4 * count1) if count1 != 0 else 0
    w2_new = w2 * total / (4 * count2) if count2 != 0 else 0
    w3_new = w3 * total / (4 * count3) if count3 != 0 else 0

    weights = torch.tensor([w0_new, w1_new, w2_new, w3_new])
    return weights


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        stdv = np.sqrt(2 / m.weight.size(1))
        m.weight.data.normal_(m.weight, std=stdv)


############################### inference tools ################################


# function to run a test sample (i.e., a test dataset) in the UNet
def get_preds(
    network, test_dataset, compute_loss, device, criterion=None, detect_nan=False
):

    with torch.no_grad():
        # check if function parameters are correct
        if compute_loss:
            assert criterion is not None, "provide criterion if computing loss"

        if test_dataset.inference == "overlap":
            assert (
                test_dataset.duration - test_dataset.step
            ) % 2 == 0, "(duration-step) is not even"
            half_overlap = (test_dataset.duration - test_dataset.step) // 2
            # to re-build videos from chunks

            # adapt half_overlap duration if using temporal reduction
            if test_dataset.temporal_reduction:
                assert half_overlap % test_dataset.num_channels == 0, (
                    "with temporal reduction half_overlap must be "
                    "a multiple of num_channels"
                )

                half_overlap_mask = half_overlap // test_dataset.num_channels
            else:
                half_overlap_mask = half_overlap

        xs = []
        ys = []
        preds = []
        n_chunks = len(test_dataset)

        for i, (x, y) in enumerate(test_dataset):
            # define start and end of used frames in chunks
            start = 0 if i == 0 else half_overlap
            start_mask = 0 if i == 0 else half_overlap_mask
            end = None if i + 1 == n_chunks else -half_overlap
            end_mask = None if i + 1 == n_chunks else -half_overlap_mask

            xs.append(x[start:end])
            ys.append(y[start_mask:end_mask])

            x = x.to(device)  # torch.cuda.FloatTensor, d x 64 x 512
            y = y[None].to(device)  # y is torch.ByteTensor
            # print("X SHAPE", x.shape)
            # print("Y SHAPE", y.shape, y.dtype)

            # detect nan in tensors
            if detect_nan:
                if torch.isnan(x).any() or torch.isnan(y).any():
                    logger.info(
                        f"Detect nan in network input (test): " "{torch.isnan(x).any()}"
                    )
                    logger.info(
                        f"Detect nan in network annotation "
                        "(test): {torch.isnan(y).any()}"
                    )

            pred = network(x[None, None])  # 1 x 4 x d x 64 x 512
            pred = pred[0]
            preds.append(pred[:, start_mask:end_mask].cpu())

        # concatenated frames and predictions for a single video:
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        preds = torch.cat(preds, dim=1)

        # print("MASK OUTPUT SHAPE BEFORE REMOVING PADDING", ys.shape)
        # print("MASK PADDING", test_dataset.pad)
        # print("REMOVED FRAMES", test_dataset.pad // test_dataset.num_channels)

        if test_dataset.pad != 0:
            start_pad = test_dataset.pad // 2
            end_pad = -(test_dataset.pad // 2 + test_dataset.pad % 2)

            xs = xs[start_pad:end_pad]

            if test_dataset.temporal_reduction:
                start_pad = start_pad // test_dataset.num_channels
                end_pad = end_pad // test_dataset.num_channels

            ys = ys[start_pad:end_pad]
            preds = preds[:, start_pad:end_pad]

        # If original sample was shorter than current movie duration, remove
        # additional padded frames
        if test_dataset.movie_duration < xs.shape[0]:
            pad = xs.shape[0] - test_dataset.movie_duration
            start_pad = pad // 2
            end_pad = -(pad // 2 + pad % 2)

            xs = xs[start_pad:end_pad]

            if test_dataset.temporal_reduction:
                start_pad = start_pad // test_dataset.num_channels
                end_pad = end_pad // test_dataset.num_channels

            ys = ys[start_pad:end_pad]
            preds = preds[:, start_pad:end_pad]

        # predictions have logarithmic values
        # print("INPUT SHAPE", xs.shape)
        # print("MASK SHAPE", ys.shape)
        # print("OUTPUT SHAPE", preds.shape)

        if compute_loss:
            loss = criterion(preds[None, :], ys[None, :]).item()
            return xs.numpy(), ys.numpy(), preds.numpy(), loss
        else:
            return xs.numpy(), ys.numpy(), preds.numpy()


def run_samples_in_model(network, device, datasets, ignore_frames):
    """
    Process al movies in the UNet and get all predictions and movies as numpy
    arrays.
    """
    network.eval()

    if hasattr(datasets[0], "video_name"):
        xs_all_videos = {}
        ys_all_videos = {}
        preds_all_videos = {}
    else:
        xs_all_videos = []
        ys_all_videos = []
        preds_all_videos = []

    for test_dataset in datasets:

        # run sample in UNet
        xs, ys, preds = get_preds(
            network=network,
            test_dataset=test_dataset,
            compute_loss=False,
            device=device,
        )

        # if dataset has video_name attribute, save results as dictionaries
        if hasattr(test_dataset, "video_name"):
            xs_all_videos[test_dataset.video_name] = xs
            ys_all_videos[test_dataset.video_name] = ys
            preds_all_videos[test_dataset.video_name] = preds
        else:
            xs_all_videos.append(xs)
            ys_all_videos.append(ys)
            preds_all_videos.append(preds)

    return xs_all_videos, ys_all_videos, preds_all_videos


################################ test function #################################


def test_function(
    network,
    device,
    criterion,
    ignore_frames,
    testing_datasets,
    # logger,
    wandb_log,
    training_name,
    output_dir,
    # training_mode=True, TODO: update code to use this (if true compute few metrics)
):
    r"""
    Validate UNet during training.
    Output segmentation is computed using argmax values and Otsu threshold (to
    remove artifacts & avoid using thresholds).
    Removing small events prior metrics computation as well.

    network:            the model being trained
    device:             current device
    criterion:          loss function to be computed on the validation set
    testing_datasets:   list of SparkDataset instances
    #logger:             logger to output results in the terminal
    ignore_frames:      frames ignored by the loss function
    wandb_log:          logger to store results on wandb
    training_name:      training name used to save predictions on disk
    output_dir:         directory where the predicted movies are saved
    training_mode:      if True, compute a smaller set of metrics (only the ones
                        that are interesting to see during training)

    """
    start = time.time()
    network.eval()

    # initialize dicts that will contain the results, indexed by movie names
    confusion_matrix = {}
    matched_events = {}
    ignored_preds = {}
    unmatched_events = {}

    # initialize class attributes that are shared by all datasets

    ca_release_events = ["sparks", "puffs", "waves"]

    sparks_type = testing_datasets[0].sparks_type
    temporal_reduction = testing_datasets[0].temporal_reduction
    if temporal_reduction:
        num_channels = testing_datasets[0].num_channels

    # spark instances detection parameters
    min_dist_xy = testing_datasets[0].min_dist_xy
    min_dist_t = testing_datasets[0].min_dist_t
    radius = math.ceil(min_dist_xy / 2)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = x**2 + y**2 <= radius**2
    conn_mask = np.stack([disk] * (min_dist_t), axis=0)

    # TODO: use better parameters !!!
    pixel_size = 0.2
    spark_min_width = 3
    spark_min_t = 3
    puff_min_t = 5
    wave_min_width = round(15 / pixel_size)

    sigma = 3

    # connectivity for event instances detection
    connectivity = 26

    # maximal gap between two predicted puffs or waves that belong together
    max_gap = 2  # i.e., 2 empty frames

    # parameters for correspondence computation
    # threshold for considering annotated and pred ROIs a match
    iomin_t = 0.5

    # compute loss on all samples
    sum_loss = 0.0

    # concatenate annotations and preds to compute segmentation-based metrics
    ys_concat = []
    preds_concat = []

    logger.debug(f"Time to load set up test function: {time.time() - start:.2f} s")

    for test_dataset in testing_datasets:

        ########################## run sample in UNet ##########################

        start = time.time()
        # get video name
        video_name = test_dataset.video_name

        # run sample in UNet
        xs, ys, preds, loss = get_preds(
            network=network,
            test_dataset=test_dataset,
            compute_loss=True,
            device=device,
            criterion=criterion,
            detect_nan=False,
        )
        # preds is a list [sparks, waves, puffs]

        # sum up losses of each sample
        sum_loss += loss

        # compute exp of predictions
        preds = np.exp(preds)

        # save preds as videos
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

        ####################### re-organise annotations ########################

        start = time.time()
        # ys_instances is a dict with classified event instances, for each class
        ys_instances = get_event_instances_class(
            event_instances=test_dataset.events, class_labels=ys, shift_ids=True
        )

        # get annotations as a dictionary
        ys_classes = {
            "sparks": np.where(ys == 1, 1, 0),
            "puffs": np.where(ys == 3, 1, 0),
            "waves": np.where(ys == 2, 1, 0),
        }

        # get pixels labelled with 4
        # TODO: togliere ignored frames ??????????????????????????????
        ignore_mask = np.where(ys == 4, 1, 0)

        ######################### get processed output #########################

        # get predicted segmentation and event instances
        preds_instances, preds_segmentation, sparks_loc = get_processed_result(
            sparks=preds[0],
            puffs=preds[2],
            waves=preds[1],
            xs=xs,
            conn_mask=conn_mask,
            connectivity=connectivity,
            max_gap=max_gap,
            sigma=sigma,
            wave_min_width=wave_min_width,
            puff_min_t=puff_min_t,
            spark_min_t=spark_min_t,
            spark_min_width=spark_min_width,
        )

        ##################### stack ys and segmented preds #####################

        # stack annotations and remove marginal frames
        ys_concat.append(empty_marginal_frames(video=ys, n_frames=ignore_frames))

        # stack preds and remove marginal frames
        temp_preds = np.zeros_like(ys)
        for event_type in ca_release_events:
            class_id = class_to_nb(event_type)
            temp_preds += class_id * preds_segmentation[event_type]
        preds_concat.append(
            empty_marginal_frames(video=temp_preds, n_frames=ignore_frames)
        )

        logger.debug(f"Time to process predictions: {time.time() - start:.2f} s")

        ############### compute pairwise scores (based on IoMin) ###############

        start = time.time()

        iomin_scores = get_score_matrix(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            ignore_mask=None,
            score="iomin",
        )

        logger.debug(f"Time to compute pairwise scores: {time.time() - start:.2f} s")

        ######################### get confusion matrix #########################

        start = time.time()

        confusion_matrix_res = get_confusion_matrix(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            scores=iomin_scores,
            t=iomin_t,
            ignore_mask=ignore_mask,
        )

        # confusion matrix cols and rows indices: {background, sparks, puffs, waves}
        confusion_matrix[video_name] = confusion_matrix_res[0]

        # dict indexed by predicted event indices, s.t. each entry is the
        # list of annotated event ids that match the predicted event
        matched_events[video_name] = confusion_matrix_res[1]

        # count number of predicted events that are ignored
        ignored_preds[video_name] = confusion_matrix_res[2]

        # get false negative (i.e., labelled but not detected) events
        unmatched_events[video_name] = confusion_matrix_res[3]

        logger.debug(f"Time to compute confusion matrix: {time.time() - start:.2f} s")

    ############################## reduce metrics ##############################

    start = time.time()

    metrics = {}

    # Compute average validation loss
    metrics["validation_loss"] = sum_loss / len(testing_datasets)
    # logger.info(f"\tvalidation loss: {loss:.4g}")

    ##################### compute instances-based metrics ######################

    """
    Metrics that can be computed (event instances):
    - Confusion matrix
    - Precision & recall (TODO)
    - F-score (e.g. beta = 0.5,1,2) (TODO)
    (- Matthews correlation coefficient (MCC))??? (TODO)
    """

    # get confusion matrix of all summed events
    metrics["events confusion matrix"] = sum(confusion_matrix.values())

    #################### compute segmentation-based metrics ####################

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

    # concatenate annotations and preds
    ys_concat = np.concatenate(ys_concat, axis=0)
    preds_concat = np.concatenate(preds_concat, axis=0)

    # concatenate ignore masks
    ignore_concat = ys_concat == 4

    # compute IoU for each class
    classes_iou = {}

    for event_type in ca_release_events:
        class_id = class_to_nb(event_type)
        class_preds = preds_concat == class_id
        class_ys = ys_concat == class_id

        metrics[event_type + "segmentation IoU"] = compute_iou(
            ys_roi=class_ys, preds_roi=class_preds, ignore_mask=ignore_concat
        )

    # compute confusion matrix
    metrics["segmentation confusion matrix"] = confusion_matrix(
        y_true=ys_concat.flatten(), y_pred=preds_concat.flatten(), labels=[0, 1, 2, 3]
    )

    if wandb_log:
        wandb.log(metrics)

    logger.debug(f"Time to reduce metrics: {time.time() - start:.2f} s")

    return metrics


# def test_function_OLD(
#     network,
#     device,
#     criterion,
#     ignore_frames,
#     testing_datasets,
#     logger,
#     wandb_log,
#     training_name,
#     output_dir,
#     training_mode=True,
# ):
#     """
#     Validate UNet during training.
#     Output segmentation is computed using argmax values (to avoid using
#     thresholds).
#     Not using minimal radius to remove small events (yet).

#     network:            the model being trained
#     device:             current device
#     criterion:          loss function to be computed on the validation set
#     testing_datasets:   list of SparkDataset instances
#     logger:             logger to output results in the terminal
#     ignore_frames:      frames ignored by the loss function
#     wandb_log:          logger to store results on wandb
#     training_name:      training name used to save predictions on disk
#     output_dir:         directory where the predicted movies are saved
#     training_mode:      if True, compute a smaller set of metrics (only the ones
#                         that are interesting to see during training)

#     """
#     network.eval()

#     pixel_based_results = {}  # store metrics for each video
#     pixel_based_results["sparks"] = {}  # movie name x {tp, tn, fp, fn}
#     pixel_based_results["puffs"] = {}  # movie name x {tp, tn, fp, fn}
#     pixel_based_results["waves"] = {}  # movie name x {tp, tn, fp, fn}

#     spark_peaks_results = {}  # store metrics for each video
#     spark_peaks_results["sparks"] = {}  # movie name x {tp, tp_fp, tp_fn}

#     # initialize class attributes that are shared by all datasets
#     sparks_type = testing_datasets[0].sparks_type
#     temporal_reduction = testing_datasets[0].temporal_reduction
#     if temporal_reduction:
#         num_channels = testing_datasets[0].num_channels

#     min_dist_xy = testing_datasets[0].min_dist_xy
#     min_dist_t = testing_datasets[0].min_dist_t

#     for test_dataset in testing_datasets:
#         # run sample in UNet
#         xs, ys, preds, loss = get_preds(
#             network=network,
#             test_dataset=test_dataset,
#             compute_loss=True,
#             device=device,
#             criterion=criterion,
#             detect_nan=False,
#         )
#         # preds is a list [sparks, waves, puffs]

#         # save preds as videos
#         write_videos_on_disk(
#             xs=xs,
#             ys=ys,
#             preds=preds,
#             training_name=training_name,
#             video_name=test_dataset.video_name,
#             path=output_dir,
#         )

#         ################## compute metrics for current video ###################

#         # get annotations as a dictionary
#         ys_classes = {
#             "sparks": np.where(ys == 1, 1, 0),
#             "puffs": np.where(ys == 3, 1, 0),
#             "waves": np.where(ys == 2, 1, 0),
#         }

#         # get pixels labelled with 4
#         # TODO: togliere ignored frames ??????????????????????????????
#         ignore_mask = np.where(ys == 4, 1, 0)

#         # compute exp of predictions
#         preds = np.exp(preds)

#         # get segmented output using argmax (as a dict)
#         argmax_preds = get_argmax_segmented_output(preds=preds, get_classes=True)[0]

#         # get list of classes
#         classes_list = argmax_preds.keys()

#         for event_class in classes_list:

#             ################## COMPUTE PIXEL-BASED RESULTS #####################

#             class_preds = argmax_preds[event_class]
#             class_ys = ys_classes[event_class]

#             # can remove marginal frames, since considering pixel-based metrics
#             class_preds = empty_marginal_frames(class_preds, ignore_frames)
#             class_ys = empty_marginal_frames(class_ys, ignore_frames)

#             tp, tn, fp, fn = compute_puff_wave_metrics(
#                 ys=class_ys,
#                 preds=class_preds,
#                 exclusion_radius=0,
#                 ignore_mask=ignore_mask,
#                 sparks=(event_class == "sparks"),
#                 results_only=True,
#             )
#             # dict with keys 'iou', 'prec', 'rec' and accuracy

#             pixel_based_results[event_class][test_dataset.video_name] = {
#                 "tp": tp,
#                 "tn": tn,
#                 "fp": fp,
#                 "fn": fn,
#             }

#             ################### COMPUTE SPARK PEAKS RESULTS ####################

#             if event_class == "sparks":
#                 # get sparks preds
#                 class_preds = argmax_preds[event_class]

#                 # get predicted peaks locations
#                 connectivity_mask = sparks_connectivity_mask(min_dist_xy, min_dist_t)
#                 coords_pred = simple_nonmaxima_suppression(
#                     img=xs,
#                     maxima_mask=class_preds,
#                     min_dist=connectivity_mask,
#                     return_mask=False,
#                     threshold=0,
#                     sigma=2,
#                 )  # TODO: maybe need to change sigma!!!!!

#                 # remove events ignored by loss function in preds
#                 if ignore_frames > 0:
#                     mask_duration = ys.shape[0]
#                     coords_pred = empty_marginal_frames_from_coords(
#                         coords=coords_pred,
#                         n_frames=ignore_frames,
#                         duration=mask_duration,
#                     )
#                     coords_true = empty_marginal_frames_from_coords(
#                         coords=test_dataset.coords_true,
#                         n_frames=ignore_frames,
#                         duration=mask_duration,
#                     )

#                 # get results as a dict {tp, tp_fp, tp_fn}
#                 res = correspondences_precision_recall(
#                     coords_real=coords_true,
#                     coords_pred=coords_pred,
#                     match_distance_xy=min_dist_xy,
#                     match_distance_t=min_dist_t,
#                     return_pairs_coords=True,
#                     return_nb_results=True,
#                 )

#                 (
#                     nb_results,
#                     paired_real,
#                     paired_pred,
#                     false_positives,
#                     false_negatives,
#                 ) = res

#                 spark_peaks_results[event_class][test_dataset.video_name] = nb_results

#                 # write videos with colored sparks on disk
#                 write_colored_sparks_on_disk(
#                     training_name=training_name,
#                     video_name=test_dataset.video_name,
#                     paired_real=paired_real,
#                     paired_pred=paired_pred,
#                     false_positives=false_positives,
#                     false_negatives=false_negatives,
#                     path=output_dir,
#                     xs=xs,
#                 )

#     ############################## reduce metrics ##############################
#     metrics = {}

#     # Compute average validation loss
#     loss /= len(testing_datasets)

#     metrics["validation_loss"] = loss
#     # logger.info(f"\tvalidation loss: {loss:.4g}")

#     for event_class in classes_list:

#         #################### COMPUTE PIXEL-BASED RESULTS #######################
#         """
#         Metrics that can be computed (raw sparks, puffs, waves):
#         - Jaccard index (IoU)
#         - Dice score
#         - Precision & recall
#         - F-score (e.g. beta = 0.5,1,2)
#         - Accuracy (biased since background is predominant)
#         - Matthews correlation coefficient (MCC)
#         """

#         # compute average of all results
#         res = {}  # tp, tn, fp, fn
#         for movie_name, movie_res in pixel_based_results[event_class].items():
#             for r, val in movie_res.items():
#                 if r in res:
#                     res[r] += val / len(testing_datasets)
#                 else:
#                     res[r] = val / len(testing_datasets)

#         # during training, compute only iou, prec & rec for puffs & waves, and
#         # compute only prec & rec for sparks
#         if not training_mode:
#             dice = (
#                 res["tp"] / (2 * res["tp"] + res["fn"] + res["fp"])
#                 if (res["tp"] + res["fn"] + res["fp"]) != 0
#                 else 1.0
#             )
#             accuracy = (res["tp"] + res["tn"]) / (
#                 res["tp"] + res["tn"] + res["fp"] + res["fn"]
#             )
#             mcc = (
#                 (res["tp"] * res["tn"] - res["fp"] * res["fn"])
#                 / np.sqrt(
#                     (res["tp"] + res["fp"])
#                     * (res["tp"] + res["fn"])
#                     * (res["tn"] + res["fp"])
#                     * (res["tn"] + res["fn"])
#                 )
#                 if (res["tp"] + res["fp"])
#                 * (res["tp"] + res["fn"])
#                 * (res["tn"] + res["fp"])
#                 * (res["tn"] + res["fn"])
#                 != 0
#                 else 0.0
#             )
#             metrics[event_class + "/dice"] = dice
#             metrics[event_class + "/accuracy"] = accuracy
#             metrics[event_class + "/mcc"] = mcc

#             if event_class == "sparks":
#                 iou = (
#                     res["tp"] / (res["tp"] + res["fn"] + res["fp"])
#                     if (res["tp"] + res["fn"] + res["fp"]) != 0
#                     else 1.0
#                 )
#                 metrics[event_class + "/iou"] = iou

#         if event_class != "sparks":
#             iou = (
#                 res["tp"] / (res["tp"] + res["fn"] + res["fp"])
#                 if (res["tp"] + res["fn"] + res["fp"]) != 0
#                 else 1.0
#             )
#             metrics[event_class + "/iou"] = iou

#         prec = (
#             res["tp"] / (res["tp"] + res["fp"]) if (res["tp"] + res["fp"]) != 0 else 1.0
#         )
#         rec = (
#             res["tp"] / (res["tp"] + res["fn"]) if (res["tp"] + res["fn"]) != 0 else 1.0
#         )
#         metrics[event_class + "/pixel_prec"] = prec
#         metrics[event_class + "/pixel_rec"] = rec

#         ################### COMPUTE SPARK PEAKS RESULTS ####################

#         """
#         Metrics that can be computed (spark peaks):
#         - Precision & recall
#         - F-score (e.g. beta = 0.5,1,2)
#         (- Matthews correlation coefficient (MCC))???
#         """

#         if event_class == "sparks":

#             # compute average of all results
#             res = {}  # tp, tp_fn, tp_fp
#             for movie_name, movie_res in spark_peaks_results[event_class].items():
#                 for r, val in movie_res.items():
#                     if r in res:
#                         res[r] += val / len(testing_datasets)
#                     else:
#                         res[r] = val / len(testing_datasets)

#             prec = res["tp"] / res["tp_fp"] if res["tp_fp"] != 0 else 1.0
#             rec = res["tp"] / res["tp_fn"] if res["tp_fn"] != 0 else 1.0

#             metrics[event_class + "/precision"] = prec
#             metrics[event_class + "/recall"] = rec

#             # during training compute only f_1 score for spark peaks
#             betas = [0.5, 1, 2] if not training_mode else [1]
#             for beta in betas:
#                 f_score = compute_f_score(prec, rec, beta)
#                 metrics[event_class + f"/f{beta}_score"] = f_score

#     for metric, val in metrics.items():
#         logger.info(f"\t{metric}: {val:.4g}")

#     if wandb_log:
#         wandb.log(metrics)

#     return metrics
