'''
This script will contain methods useful for processing the unet outputs
'''
from collections import namedtuple

import numpy as np
import cc3d
from scipy import ndimage as ndi
from scipy import optimize, spatial
from skimage import morphology
from sklearn.metrics import roc_auc_score
from bisect import bisect_left

import imageio
import os


__all__ = ["Metrics",
           "nonmaxima_suppression",
           "process_spark_prediction",
           "inverse_argwhere",
           "correspondences_precision_recall",
           "reduce_metrics",
           "empty_marginal_frames",
           "write_videos_on_disk",
           "compute_prec_rec",
           "reduce_metrics_thresholds",
           "take_closest",
           "process_spark_prediction",
           "process_puff_prediction",
           "process_wave_prediction",
           "jaccard_score_exclusion_zone"
           ]


################################ Generic utils #################################


def empty_marginal_frames(video, n_frames):
    # Set first and last n_frames of a video to zero
    if n_frames != 0:
        new_video = video[n_frames:-n_frames]
        new_video = np.pad(new_video,((n_frames,),(0,),(0,)), mode='constant')
    else: new_video = video

    assert(np.shape(video) == np.shape(new_video))

    return new_video


def write_videos_on_disk(training_name, video_name, path="predictions",
                         xs=None, ys=None, preds=None):
    # Write all videos on disk
    # xs : input video used by network
    # ys: segmentation video used in loss function
    # preds : all u-net preds [bg preds, sparks preds, puffs preds, waves preds]

    out_name_root = training_name + "_" + video_name + "_"

    if not isinstance(xs, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "xs.tif"),
                                      xs)
    if not isinstance(ys, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "ys.tif"),
                                      ys)
    if not isinstance(preds, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "preds.tif"),
                                      np.exp(preds))


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before


################################ Sparks metrics ################################

'''
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
'''

Metrics = namedtuple('Metrics', ['precision', 'recall', 'tp', 'tp_fp', 'tp_fn'])


def in_bounds(points, shape):

    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
                                for coords_i, shape_i in zip(points.T, shape)])


def nonmaxima_suppression(img, return_mask=False,
                          neighborhood_radius=5, threshold=0.5):

    smooth_img = ndi.gaussian_filter(img, 2) # 2 instead of 1
    dilated = ndi.grey_dilation(smooth_img, (neighborhood_radius,) * img.ndim)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def get_sparks_locations_from_mask(mask, ignore_frames=0):
    '''
    Get sparks coords from annotations mask.

    mask : annotations mask (values 0,1,2,3,4)
    ignore_frames: number of frames ignored by loss fct during training
    '''

    sparks_mask = np.where(mask == 1, 1.0, 0.0)
    sparks_mask = sparks_mask[ignore_frames:-ignore_frames]
    sparks_mask = np.pad(sparks_mask,((ignore_frames,),(0,),(0,)),
                         mode='constant')

    assert(np.shape(sparks_mask) == np.shape(mask))

    coords = nonmaxima_suppression(sparks_mask)

    return coords


def process_spark_prediction(pred, t_detection = 0.9,
                             neighborhood_radius = 5,
                             min_radius = 3,
                             return_mask = False,
                             return_clean_pred = False,
                             ignore_frames = 0):
    '''
    Get sparks centres from preds: remove small events + nonmaxima suppression

    pred: network's sparks predictions
    neighborhood_radius: ??
    min_radius: minimal 'radius' of a valid spark
    return_mask: if True return mask and locations of sparks
    return_clean_pred: if True only return preds without small events
    ignore_frames: set preds in region ignored by loss fct to 0
    '''

    # set frames ignored by loss fct to 0
    pred_sparks = empty_marginal_frames(pred, ignore_frames)

    # remove small objects
    min_size = (2 * min_radius) ** pred.ndim

    pred_boolean = pred_sparks > t_detection
    small_objs_removed = morphology.remove_small_objects(pred_boolean,
                                                         min_size=min_size)
    big_pred = np.where(small_objs_removed, pred_sparks, 0)

    if return_clean_pred:
        return big_pred

    gaussian = ndi.gaussian_filter(big_pred, 2)
    dilated = ndi.grey_dilation(gaussian, (neighborhood_radius,)*pred.ndim)

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


def reduce_metrics(results):

    tp = sum(i.tp for i in results)
    tp_fp = sum(i.tp_fp for i in results)
    tp_fn = sum(i.tp_fn for i in results)

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 1.0
    if tp_fn > 0:
        recall = tp / tp_fn
    else:
        recall = 1.0

    return Metrics(precision, recall, tp, tp_fp, tp_fn)


def compute_prec_rec(annotations, preds, thresholds, ignore_frames=0,
                     min_radius=3, match_distance=6):
    # annotations: video of sparks segmentation w/ values in {0,1}
    # preds: video of sparks preds w/ values in [0,1]
    # thresholds : list of thresholds applied to the preds over which events are kept
    # min_radius : minimal "radius" of a valid event
    # match_distance : maximal distance between annotation and pred
    # returns a list of Metrics tuples corresponding to thresholds and AUC

    if ignore_frames != 0:
        annotations = empty_marginal_frames(annotations, ignore_frames)
        preds = empty_marginal_frames(preds, ignore_frames)

    metrics = {} # list of 'Metrics' tuples: precision, recall, tp, tp_fp, tp_fn
                 # indexed by threshold value

    coords_true = nonmaxima_suppression(annotations)

    # compute prec and rec for every threshold
    for t in thresholds:
        coords_preds = process_spark_prediction(preds,
                                                t_detection=t,
                                                min_radius=min_radius)

        prec_rec = Metrics(*correspondences_precision_recall(coords_true,
                                                             coords_preds,
                                                             match_distance))

        metrics[t] = prec_rec
        #print("threshold", t)
        #prec.append(prec_rec.precision)
        #rec.append(prec_rec.recall)


    # compute AUC for this sample
    #area_under_curve = auc(rec, prec)

    return metrics#, area_under_curve

def reduce_metrics_thresholds(results):
    # apply metrics reduction to results corresponding to different thresholds
    # results is a list of dicts
    # thresholds is the list of used thresholds
    # returns dicts of reduced 'Metrics' instances for every threshold

    # list of dicts to dict of lists
    results_t = {k: [dic[k] for dic in results] for k in results[0]}

    reduced_metrics = {}
    prec = {}
    rec = {}

    for t, res in results_t.items():
        # res is a list of 'Metrics' of all videos wrt a threshold
        reduced_res = reduce_metrics(res)

        reduced_metrics[t] = reduced_res
        prec[t] = reduced_res.precision
        rec[t] = reduced_res.recall

    # compute area under the curve for reduced metrics
    #print("REC",rec)
    #print("PREC",prec)
    #area_under_curve = roc_auc_score(rec, prec)
    #print("AREA UNDER CURVE", area_under_curve)

    return reduced_metrics, prec, rec, None



############################ Puffs and waves metrics ###########################

'''
Utils for computing metrics related to puffs and waves, e.g.:
- Jaccard index
- exclusion region for Jaccard index
'''

def separate_events(pred, t_detection=0.5, min_radius=4):
    '''
    Apply threshold to prediction and separate the events (1 event = 1 connected
    component).
    '''
    # apply threshold to prediction
    pred_boolean = pred >= t_detection

    # clean events
    min_size = (2 * min_radius) ** pred.ndim
    pred_clean = morphology.remove_small_objects(pred_boolean,
                                                 min_size=min_size)
    #big_pred = np.where(small_objs_removed, pred, 0)

    # separate events
    connectivity = 26
    labels, n_events = cc3d.connected_components(pred_clean,
                                                 connectivity=connectivity,
                                                 return_N=True)

    return labels, n_events


def process_puff_prediction(pred, t_detection = 0.5,
                            min_radius = 4,
                            ignore_frames = 0):
    '''
    Get binary clean predictions of puffs (remove small preds)

    pred: network's puffs predictions
    min_radius : minimal 'radius' of a valid puff
    ignore_frames: set preds in region ignored by loss fct to 0
    '''

    # set first and last frames to 0 according to ignore_frames
    if ignore_frames != 0:
        pred_puffs = empty_marginal_frames(pred, ignore_frames)

    # remove small objects
    min_size = (2 * min_radius) ** pred.ndim

    pred_boolean = pred_puffs > t_detection
    small_objs_removed = morphology.remove_small_objects(pred_boolean,
                                                         min_size=min_size)

    #big_pred = np.where(small_objs_removed, pred_puffs, 0) # not binary version


    return small_objs_removed


def process_wave_prediction(pred, t_detection = 0.5,
                            min_radius = 4,
                            ignore_frames = 0):

    # for now: do the same as with puffs

    return process_puff_prediction(pred, t_detection, min_radius, ignore_frames)


def jaccard_score_exclusion_zone(ys,preds,exclusion_radius,ignore_mask=None,sparks=False):
    # ys, preds and ignore_mask are binary masks

    # Compute intersection and union
    intersection = np.logical_and(ys, preds)
    union = np.logical_or(ys, preds)

    if exclusion_radius != 0:
        # Compute exclusion zone: 1 where Jaccard index has to be computed, 0 otherwise
        dilated = ndi.binary_dilation(ys, iterations=exclusion_radius)

        if not sparks:
            eroded = ndi.binary_erosion(ys, iterations=exclusion_radius)
            exclusion_mask = 1 - np.logical_xor(eroded,dilated)
        else:
            # Erosion is not computed for spark class
            exclusion_mask = 1 - np.logical_xor(ys,dilated)

        # If ignore mask is given, don't compute values where it is 1
        if ignore_mask is not None:
            # Compute dilation for ignore mask too (erosion not necessary)
            ignore_mask = ndi.binary_dilation(ignore_mask, iterations=exclusion_radius)

            # Ignore regions where ignore mask is 1
            exclusion_mask = np.logical_and(1 - ignore_mask, exclusion_mask)

        # Compute intersecion of exclusion mask with intersection and union
        intersection = np.logical_and(intersection, exclusion_mask)
        union = np.logical_and(union, exclusion_mask)

    #print("Pixels in intersection:", np.count_nonzero(intersection))
    #print("Pixels in union:", np.count_nonzero(union))

    if np.count_nonzero(union) != 0:
        iou = np.count_nonzero(intersection)/np.count_nonzero(union)
    else:
        iou = 1.

    return iou
