
from collections import namedtuple

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize, spatial
from skimage import morphology


__all__ = ["Metrics",
           "nonmaxima_suppression",
           "process_spark_prediction",
           "inverse_argwhere",
           "correspondences_precision_recall",
           "new_correspondences_precision_recall",
           "reduce_metrics"]

Metrics = namedtuple('Metrics', ['precision', 'recall', 'tp', 'tp_fp', 'tp_fn'])


def in_bounds(points, shape):

    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
                                for coords_i, shape_i in zip(points.T, shape)])


def nonmaxima_suppression(img, return_mask=False, neighborhood_radius=5, threshold=0.5):

    smooth_img = ndi.gaussian_filter(img, 2) # 2 instead of 1
    dilated = ndi.grey_dilation(smooth_img, (neighborhood_radius,) * img.ndim)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def process_spark_prediction(pred, t_detection = 0.9, neighborhood_radius = 5, min_radius = 4, return_mask = False, return_clean_pred = False):
    # remove small objects
    min_size = (2 * min_radius) ** pred.ndim

    pred_boolean = pred > t_detection
    small_objs_removed = morphology.remove_small_objects(pred_boolean, min_size=min_size)
    big_pred = np.where(small_objs_removed, pred, 0)

    if return_clean_pred:
        return big_pred

    gaussian = ndi.gaussian_filter(big_pred, 2)
    dilated = ndi.grey_dilation(gaussian, (neighborhood_radius,) * pred.ndim)

    # detect events (nonmaxima suppression)
    argmaxima = np.logical_and(gaussian == dilated, big_pred > t_detection)
    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def inverse_argwhere(coords, shape, dtype):
    """
    Creates an array with given shape and dtype such that

    np.argwhere(inverse_argwhere(coords, shape, dtype)) == coords

    up to a rounding of `coords`.
    """

    res = np.zeros(shape, dtype=dtype)
    intcoords = np.int_(np.round(coords))
    intcoords = intcoords[in_bounds(intcoords, shape)]
    res[intcoords[:, 0], intcoords[:, 1], intcoords[:, 2]] = 1
    return res


def correspondences_precision_recall(coords_real, coords_pred, match_distance):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.
    """

    w = spatial.distance_matrix(coords_real, coords_pred)
    w[w > match_distance] = 9999999 # NEW
    row_ind, col_ind = optimize.linear_sum_assignment(w)

    tp = np.count_nonzero(w[row_ind, col_ind] <= match_distance)
    tp_fp = len(coords_pred)
    tp_fn = len(coords_real)

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 1.0

    if tp_fn > 0:
        recall = tp / tp_fn
    else:
        recall = 1.0

    return precision, recall, tp, tp_fp, tp_fn

# new function for correspondences: closest point to annotation instead of minimal weight matching

def new_correspondences_precision_recall(coords_real, coords_pred, match_distance):
    tp_fn = len(coords_real)
    tp_fp = len(coords_pred)

    if (tp_fn == 0 or tp_fp == 0):
        tp = 0
    else:
        w = spatial.distance_matrix(coords_real, coords_pred)
        #print("distance matrix shape ", np.shape(w))
        closest_annotation = np.argmin(w, axis=1)
        tp = np.count_nonzero(w[range(w.shape[0]),closest_annotation] <= match_distance)

        # TODO:
        # se due annotations vengono assegnate alla stessa
        # prediction, prendere la piu vicina

    if tp_fn == 0: # no annotations
        recall = 1
    else:
        recall = tp / tp_fn

    if tp_fp == 0: # no predictions
        precision = 1
    else:
        precision = tp / tp_fp

    return precision, recall, tp, tp_fp, tp_fn


def reduce_metrics(results):

    tp = sum(i.tp for i in results)
    tp_fp = sum(i.tp_fp for i in results)
    tp_fn = sum(i.tp_fn for i in results)

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 1.0
    recall = tp / tp_fn

    return Metrics(precision, recall, tp, tp_fp, tp_fn)
