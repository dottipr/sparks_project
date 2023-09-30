"""
Script with function for metrics on UNet output computation.

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as ..., )
16.06.2023: removed some unused functions (both in code and at the end of the script)

Author: Prisca Dotti
Last modified: 28.09.2023

"""
from bisect import bisect_left
from collections import namedtuple
import logging

import numpy as np
from scipy import optimize, spatial, sparse

from config import config

logger = logging.getLogger(__name__)

################################ Generic utils #################################


def list_difference(l1, l2):
    """
    Compute the difference between two lists, l1 - l2.

    Args:
        l1 (list): The first list.
        l2 (list): The second list.

    Returns:
        list: The elements that are in l1 but not in l2.
    """
    return [item for item in l1 if item not in l2]


############################### Generic metrics ################################


def get_metrics_from_summary(tot_preds,
                             tp_preds,
                             ignored_preds,
                             unlabeled_preds,
                             tot_ys,
                             tp_ys,
                             undetected_ys):
    """
    Compute instance-based metrics from matched events summary.

    Instance-based metrics are:
    - precision per class
    - recall per class
    - % correctly classified events per class
    - % detected events per class

    Parameters:
        tot_preds (dict): Total number of predicted events per class.
        tp_preds (dict): True positive predicted events per class.
        ignored_preds (dict): Ignored predicted events per class.
        unlabeled_preds (dict): Unlabeled predicted events per class.
        tot_ys (dict): Total number of annotated events per class.
        tp_ys (dict): True positive annotated events per class.
        undetected_ys (dict): Undetected annotated events per class.

    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {}

    for event_type in config.classes_dict.keys():
        if event_type == 'ignore':
            continue

        denom_preds = tot_preds[event_type] - ignored_preds[event_type]
        denom_ys = tot_ys[event_type]

        precision = tp_preds[event_type] / \
            denom_preds if denom_preds > 0 else 0
        recall = tp_ys[event_type] / denom_ys if denom_ys > 0 else 0
        correctly_classified = tp_preds[event_type] / (
            denom_preds - unlabeled_preds[event_type]) if denom_preds > 0 else 0
        detected = 1 - (undetected_ys[event_type] /
                        denom_ys) if denom_ys > 0 else 0

        metrics[event_type + "/precision"] = precision
        metrics[event_type + "/recall"] = recall
        metrics[event_type + "/correctly_classified"] = correctly_classified
        metrics[event_type + "/detected"] = detected

    # Compute average over classes for each metric
    for m in ["precision", "recall", "correctly_classified", "detected"]:
        metrics["average/" + m] = np.mean(
            [metrics[event_type + "/" + m]
                for event_type in config.classes_dict.keys()
                if event_type != 'ignore']
        )

    return metrics


def compute_iou(ys_roi, preds_roi, ignore_mask=None, debug=False):
    """
    Compute Intersection over Union (IoU) for given single annotated and predicted
    events.

    Args:
        ys_roi (numpy.ndarray): Annotated event ROI (binary mask).
        preds_roi (numpy.ndarray): Predicted event ROI (binary mask).
        ignore_mask (numpy.ndarray, optional): Mask that is ignored by the loss
            function during training.
        debug (bool, optional): If True, print a warning when both ys and preds are
            empty.

    Returns:
        float: The computed IoU value.
    """
    # Define a mask where pixels aren't ignored by the loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    # Calculate the intersection and union of the masks
    intersection = np.logical_and(ys_roi, preds_roi_real)
    union = np.logical_or(ys_roi, preds_roi_real)

    # Compute IoU
    if np.count_nonzero(union):
        iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
        iou = 1.0
        if debug:
            logger.warning("Warning: both annotations and preds are empty")

    return iou


def compute_inter_min(ys_roi, preds_roi, ignore_mask=None):
    """
    Compute Intersection over Minimum Area for given single annotated and predicted
    events.

    Args:
        ys_roi (numpy.ndarray): Annotated event ROI (binary mask).
        preds_roi (numpy.ndarray): Predicted event ROI (binary mask).
        ignore_mask (numpy.ndarray, optional): Mask that is ignored by the loss
            function during training.

    Returns:
        float: The computed Intersection over Minimum Area (IoMin) value.
    """
    # Define a mask where pixels aren't ignored by the loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    # Calculate the intersection and areas of the masks
    intersection = np.logical_and(ys_roi, preds_roi_real)
    ys_area = np.count_nonzero(ys_roi)
    preds_area = np.count_nonzero(preds_roi_real)

    # Compute IoMin
    if preds_area > 0:
        iomin = np.count_nonzero(intersection) / min(preds_area, ys_area)
    else:
        iomin = 0

    return iomin


def compute_iomin_one_hot(y_vector, preds_array):
    """
    Compute Intersection over Minimum Score for a given single annotated event and
    all predicted events.

    Args:
        y_vector (csr_matrix): Flattened, one-hot encoded annotated event CSR matrix.
            (shape = 1 x movie shape)
        preds_array (csr_matrix): Flattened, one-hot encoded predicted events CSR
            matrix. Ensure that predicted events are intersected with the negation of
            the ignore mask before calling this function.
            (shape = #preds x movie shape)

    Returns:
        numpy.ndarray: List of IoMin scores for each predicted event.
    """
    # Check that y_vector is not empty
    assert y_vector.count_nonzero != 0, "y_vector is empty"

    # Compute the intersection of CSR matrix y_vector with each row of CSR matrix
    # preds_array
    intersection = y_vector.multiply(preds_array)

    if intersection.count_nonzero() == 0:
        return np.zeros(preds_array.shape[0])

    else:
        # Compute non-zero elements for each row (predicted event) of intersection
        intersection_area = intersection.getnnz(axis=1).astype(float)

        # Get the denominator for IoMin using non-zero elements of preds_array and
        # y_vector
        denominator = np.minimum(preds_array.getnnz(axis=1),
                                 y_vector.getnnz()).astype(float)

        # Compute IoMin by dividing intersection_area by denominator
        # If denominator is 0, set IoMin to 0
        scores = np.divide(
            intersection_area,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float16),
            where=(denominator > 0),
        )

        return scores


################################ Sparks metrics ################################
"""
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
"""


def correspondences_precision_recall(
    coords_real,
    coords_pred,
    return_pairs_coords=False,
    return_nb_results=False,
):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.

    Args:
        coords_real (numpy.ndarray): Array of real coordinates.
        coords_pred (numpy.ndarray): Array of predicted coordinates.
        return_pairs_coords (bool): Whether to return paired coordinates.
        return_nb_results (bool): Whether to return only TP, TP+FP, TP+FN as a dict.

    Returns:
        tuple or dict: Precision, recall, F1-score, TP, TP+FP, TP+FN, paired real,
            paired pred, false positives, and false negatives based on function
            arguments.
    """
    # Divide temporal and spatial coordinates by match distances
    if coords_real.size > 0:
        coords_real[:, 0] /= config.min_dist_t
        coords_real[:, 1] /= config.min_dist_xy
        coords_real[:, 2] /= config.min_dist_xy

    if coords_pred.size > 0:
        coords_pred[:, 0] /= config.min_dist_t
        coords_pred[:, 1] /= config.min_dist_xy
        coords_pred[:, 2] /= config.min_dist_xy

    if coords_real.size and coords_pred.size > 0:
        w = spatial.distance_matrix(coords_real, coords_pred)
        w[w > 1] = 9999999  # Set a high value for distances greater than 1
        row_ind, col_ind = optimize.linear_sum_assignment(w)

    if return_pairs_coords:
        # Multiply coordinates by match distances
        if coords_real.size > 0:
            coords_real[:, 0] *= config.min_dist_t
            coords_real[:, 1] *= config.min_dist_xy
            coords_real[:, 2] *= config.min_dist_xy

        if coords_pred.size > 0:
            coords_pred[:, 0] *= config.min_dist_t
            coords_pred[:, 1] *= config.min_dist_xy
            coords_pred[:, 2] *= config.min_dist_xy

        if coords_real.size and coords_pred.size > 0:
            # True positive pairs:
            paired_real = [
                coords_real[i].tolist()
                for i, j in zip(row_ind, col_ind)
                if w[i, j] <= 1
            ]
            paired_pred = [
                coords_pred[j].tolist()
                for i, j in zip(row_ind, col_ind)
                if w[i, j] <= 1
            ]

            # False positive (predictions):
            false_positives = list_difference(coords_pred, paired_pred)

            # False negative (annotations):
            false_negatives = list_difference(coords_real, paired_real)

            if return_nb_results:
                tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}
                return res, paired_real, paired_pred, false_positives, false_negatives
            else:
                return paired_real, paired_pred, false_positives, false_negatives
        else:
            if return_nb_results:
                tp = 0
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}
                return res, [], [], coords_pred, coords_real
            else:
                return [], [], coords_pred, coords_real

    else:
        if (coords_real.size > 0) and (coords_pred.size > 0):
            tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
        else:
            tp = 0

        tp_fp = len(coords_pred)
        tp_fn = len(coords_real)

        if return_nb_results:
            return {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}

        if tp_fp > 0:
            precision = tp / tp_fp
        else:
            precision = 1.0

        if tp_fn > 0:
            recall = tp / tp_fn
        else:
            recall = 1.0

        f1_score = compute_f_score(precision, recall)

        return precision, recall, f1_score, tp, tp_fp, tp_fn


def compute_f_score(prec, rec, beta=1):
    if beta == 1:
        f_score = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0.0
    else:
        f_score = (
            (1 + beta * beta) * (prec + rec) / (beta * beta * prec + rec)
            if prec + rec != 0
            else 0.0
        )
    return f_score


########################### Instances-based metrics ############################


def _get_sparse_binary_encoded_mask(mask):
    """
    Create a sparse binary encoded mask from an array with labeled event instances.

    Args:
        mask (numpy.ndarray): Array with labeled event instances.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix with one-hot encoding of the events
        mask (each row corresponds to a different event).
    """
    # Flatten mask
    v = mask.flatten()

    # Get one-hot encoding of events as a sparse matrix
    rows = v
    cols = np.arange(v.size)
    data = np.ones(v.size, dtype=bool)
    sparse_v = sparse.csr_array(
        (data, (rows, cols)), shape=(v.max() + 1, v.size), dtype=bool
    )

    # Remove background "event"
    sparse_v = sparse_v[1:]
    return sparse_v


def get_score_matrix(ys_instances, preds_instances, ignore_mask=None):
    """
    Compute pairwise IoMin scores between annotated event instances and predicted
    event instances.

    Args:
        ys_instances (dict): Dictionary of annotated event instances, indexed by
            event types (each entry is an int array).
        preds_instances (dict): Dictionary of predicted event instances, indexed by
            event types (each entry is an int array).
        ignore_mask (numpy.ndarray, optional): Binary mask indicating ROIs ignored
            during training.

    Returns:
        numpy.ndarray: Array of shape (n_ys_events, n_preds_events) containing
        pairwise scores.
    """
    # Compute matrices with all separated events summed
    ys_all_events = sum(ys_instances.values())
    preds_all_events = sum(preds_instances.values())

    # Intersect predicted events with the negation of the ignore mask
    if ignore_mask is not None:
        preds_all_events = np.logical_and(
            preds_all_events, np.logical_not(ignore_mask)
        )

    # Convert to one-hot encoding and transpose matrices
    ys_all_events = _get_sparse_binary_encoded_mask(ys_all_events)
    preds_all_events = _get_sparse_binary_encoded_mask(preds_all_events)

    # Compute pairwise scores
    scores = []
    for y_vector in ys_all_events:
        y_scores = compute_iomin_one_hot(
            y_vector=y_vector, preds_array=preds_all_events
        )
        scores.append(y_scores)

    scores = np.array(scores)

    return scores


def get_matches_summary(ys_instances, preds_instances, scores, ignore_mask):
    """
    Analyze matched predicted events with annotated events and categorize them.

    Args:
        ys_instances (dict): Dictionary of annotated event instances.
        preds_instances (dict): Dictionary of predicted event instances.
        scores (numpy.ndarray): Array of pairwise scores.
        ignore_mask (numpy.ndarray): Binary mask, ROIs ignored during training.

    Returns:
        Tuple: A tuple containing matched_ys_ids (annotated events summary) and
        matched_preds_ids (predicted events summary).
    """
    # Initialize dicts that summarize the results
    matched_ys_ids = {ca_class: {} for ca_class in config.classes_dict.keys()
                      if ca_class != 'ignore'}
    matched_preds_ids = {ca_class: {} for ca_class in config.classes_dict.keys()
                         if ca_class != 'ignore'}

    for ca_class in config.classes_dict.keys():
        if ca_class == 'ignore':
            continue

        # Get set of IDs of all annotated events
        ys_ids = set(np.unique(ys_instances[ca_class])) - {0}
        matched_ys_ids[ca_class]['tot'] = ys_ids.copy()

        # Get set of IDs of all predicted events
        preds_ids = set(np.unique(preds_instances[ca_class])) - {0}
        matched_preds_ids[ca_class]['tot'] = preds_ids.copy()

        for other_class in config.classes_dict.keys():
            if other_class == 'ignore' or other_class == ca_class:
                continue

            # Initialize mispredicted events (in annotations and predictions)
            matched_ys_ids[ca_class][other_class] = set()
            matched_preds_ids[ca_class][other_class] = set()

        # Initialize undetected annotated events
        matched_ys_ids[ca_class]['undetected'] = ys_ids.copy()

    for ca_class in config.classes_dict.keys():
        if ca_class == 'ignore':
            continue

        # Initialize sets of correctly matched annotations and predictions
        matched_preds_ids[ca_class]['tp'] = set()
        matched_ys_ids[ca_class]['tp'] = set()

        # Initialize ignored predicted events
        matched_preds_ids[ca_class]['ignored'] = set()

        # Initialize predicted events not matched with any label
        matched_preds_ids[ca_class]['unlabeled'] = set()

        ### Go through predicted events and match them with annotated events ###
        for pred_id in matched_preds_ids[ca_class]['tot']:
            # Get set of y_ids that are matched with pred_id (score > t):
            matched_events = set(
                np.where(scores[:, pred_id - 1] >= config.iomin_t)[0] + 1)

            # If matched_events is empty, chech if pred_id is ignored
            if not matched_events:
                pred_roi = preds_instances[ca_class] == pred_id
                ignored_roi = np.logical_and(pred_roi, ignore_mask)
                overlap = np.count_nonzero(
                    ignored_roi) / np.count_nonzero(pred_roi)

                if overlap >= config.iomin_t:
                    # Mark detected event as ignored
                    matched_preds_ids[ca_class]['ignored'].add(pred_id)
                else:
                    # Detected event does not match any labelled event
                    matched_preds_ids[ca_class]['unlabeled'].add(pred_id)

            # Otherwise, pred_id matches with at least one labelled event
            else:
                for other_class in config.classes_dict.keys():
                    if other_class == 'ignore':
                        continue

                    # Check if pred_id matched with an event of the other class
                    matched_other_class = matched_events & matched_ys_ids[other_class]['tot']

                    # Remove matched events from undetected events
                    matched_ys_ids[other_class]['undetected'] -= matched_other_class

                    if matched_other_class:
                        if other_class == ca_class:
                            # pred_id is a correct prediction
                            matched_preds_ids[ca_class]['tp'].add(pred_id)
                            matched_ys_ids[ca_class]['tp'] |= matched_other_class
                        else:
                            # pred_id is misclassified
                            matched_preds_ids[ca_class][other_class].add(
                                pred_id)
                            matched_ys_ids[other_class][ca_class] |= matched_other_class

    return matched_ys_ids, matched_preds_ids
