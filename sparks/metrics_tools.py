'''
This script will contain methods useful for processing the unet outputs
'''
import glob
import imageio
import os

from collections import namedtuple, defaultdict

import numpy as np
import cc3d
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import optimize, spatial
from skimage import morphology
from skimage.draw import ellipsoid
from sklearn.metrics import auc
from bisect import bisect_left


__all__ = ["Metrics",
           "nonmaxima_suppression",
           "inverse_argwhere",
           "correspondences_precision_recall",
           "reduce_metrics",
           "empty_marginal_frames",
           "write_videos_on_disk",
           "compute_prec_rec",
           "reduce_metrics_thresholds",
           "take_closest",
           "get_sparks_locations_from_mask",
           "process_spark_prediction",
           "process_puff_prediction",
           "process_wave_prediction",
           "jaccard_score_exclusion_zone"
           ]


################################ Global params #################################

# physiological params to get sparks locations
# these have to be coherent in the whole project

PIXEL_SIZE = 0.2 # 1 pixel = 0.2 um x 0.2 um
global MIN_DIST_XY
MIN_DIST_XY = round(1.8 / PIXEL_SIZE) # min distance in space between sparks
TIME_FRAME = 6.8 # 1 frame = 6.8 ms
global MIN_DIST_T
MIN_DIST_T = round(20 / TIME_FRAME) # min distance in time between sparks


################################ Generic utils #################################


def empty_marginal_frames(video, n_frames):
    '''
    Set first and last n_frames of a video to zero.
    '''
    if n_frames != 0:
        new_video = video[n_frames:-n_frames]
        new_video = np.pad(new_video,((n_frames,),(0,),(0,)), mode='constant')
    else: new_video = video

    assert(np.shape(video) == np.shape(new_video))

    return new_video


def write_videos_on_disk(training_name, video_name, path="predictions",
                         xs=None, ys=None, preds=None):
    '''
     Write all videos on disk
     xs : input video used by network
     ys: segmentation video used in loss function
     preds : all u-net preds [bg preds, sparks preds, puffs preds, waves preds]
    '''
    out_name_root = training_name + "_" + video_name + "_"

    if not isinstance(xs, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "xs.tif"),
                                      xs)
    if not isinstance(ys, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "ys.tif"),
                                      np.uint8(ys))
    if not isinstance(preds, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "sparks.tif"),
                                      np.exp(preds[1]))
        imageio.volwrite(os.path.join(path, out_name_root + "waves.tif"),
                                      np.exp(preds[2]))
        imageio.volwrite(os.path.join(path, out_name_root + "puffs.tif"),
                                      np.exp(preds[3]))


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


def diff(l1, l2):
    # l1 and l2 are lists
    # return l1 - l2
    return list(map(list,(set(map(tuple,l1))).difference(set(map(tuple,l2)))))


def flood_fill_hull(image):
    '''
    Compute convex hull of a Numpy array.
    '''
    points = np.transpose(np.where(image))
    hull = spatial.ConvexHull(points)
    deln = spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


################################ Sparks metrics ################################

'''
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
'''

Metrics = namedtuple('Metrics', ['precision', 'recall', 'tp', 'tp_fp', 'tp_fn'])


#def in_bounds(points, shape):
#
#    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
#                                for coords_i, shape_i in zip(points.T, shape)])


def nonmaxima_suppression(img,
                          min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T,
                          return_mask=False, threshold=0.5, sigma=2):
    '''
    Extract local maxima from input array (t,x,y).
    img : input array
    min_dist_xy : minimal spatial distance between two maxima
    min_dist_t : minimal temporal distance between two maxima
    return_mask : if True return both masks with maxima and locations, if False
                  only returns locations
    threshold : minimal value of maximum points
    sigma : sigma parameter of gaussian filter
    '''

    smooth_img = ndi.gaussian_filter(img, sigma)
    #smooth_img = img

    min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
    #dilated = ndi.grey_dilation(smooth_img,
    dilated = ndi.maximum_filter(smooth_img,
                                 footprint=min_dist)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def get_sparks_locations_from_mask(mask, min_dist_xy, min_dist_t,
                                   ignore_frames=0):
    '''
    Get sparks coords from annotations mask.

    mask : annotations mask (values 0,1,2,3,4) where sparks are denoted by peaks
    ignore_frames: number of frames ignored by loss fct during training
    '''

    sparks_mask = np.where(mask == 1, 1.0, 0.0)
    sparks_mask = empty_marginal_frames(sparks_mask, ignore_frames)

    coords = nonmaxima_suppression(sparks_mask, min_dist_xy, min_dist_t)

    return coords


def process_spark_prediction(pred,
                             t_detection = 0.9,
                             min_dist_xy = MIN_DIST_XY,
                             min_dist_t = MIN_DIST_T,
                             min_radius = 3,
                             return_mask = False,
                             return_clean_pred = False,
                             ignore_frames = 0):
    '''
    Get sparks centres from preds: remove small events + nonmaxima suppression

    pred: network's sparks predictions
    t_detection: sparks detection threshold
    min_dist_xy : minimal spatial distance between two maxima
    min_dist_t : minimal temporal distance between two maxima
    min_radius: minimal 'radius' of a valid spark
    return_mask: if True return mask and locations of sparks
    return_clean_pred: if True only return preds without small events
    ignore_frames: set preds in region ignored by loss fct to 0
    '''

    # remove small objects
    min_size = (2 * min_radius) ** pred.ndim

    pred_boolean = pred > t_detection
    if min_size > 0:
        small_objs_removed = morphology.remove_small_objects(pred_boolean,
                                                             min_size=min_size)
    else:
        small_objs_removed = pred_boolean
    # oginal preds without small objects:
    big_pred = np.where(small_objs_removed, pred, 0)

    if return_clean_pred:
        big_pred = empty_marginal_frames(big_pred, ignore_frames)
        return big_pred

    # detect events (nonmaxima suppression)
    argwhere, argmaxima = nonmaxima_suppression(img=big_pred,
                                                min_dist_xy=min_dist_xy,
                                                min_dist_t=min_dist_t,
                                                return_mask=True,
                                                threshold=t_detection,
                                                sigma=2)

    if not return_mask:
        return argwhere

    # set frames ignored by loss fct to 0
    argmaxima = empty_marginal_frames(argmaxima, ignore_frames)

    return argwhere, argmaxima


#def inverse_argwhere(coords, shape, dtype):
#    """
#    Creates an array with given shape and dtype such that
#
#    np.argwhere(inverse_argwhere(coords, shape, dtype)) == coords
#
#    up to a rounding of `coords`.
#    """
#
#    res = np.zeros(shape, dtype=dtype)
#    intcoords = np.int_(np.round(coords))
#    intcoords = intcoords[in_bounds(intcoords, shape)]
#    res[intcoords[:, 0], intcoords[:, 1], intcoords[:, 2]] = 1
#    return res


def correspondences_precision_recall(coords_real, coords_pred,
                                     match_distance_t = MIN_DIST_T,
                                     match_distance_xy = MIN_DIST_XY,
                                     return_pairs_coords = False):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.

    If return_pairs_coords == True, return paired sparks coordinated
    """
    # convert coords to arrays
    coords_real = np.asarray(coords_real, dtype=float)
    coords_pred = np.asarray(coords_pred, dtype=float)

    # divide temporal coords by match_distance_t and spatial coords by
    # match_distance_xy
    coords_real[:,0] /= match_distance_t
    coords_real[:,1] /= match_distance_xy
    coords_real[:,2] /= match_distance_xy

    coords_pred[:,0] /= match_distance_t
    coords_pred[:,1] /= match_distance_xy
    coords_pred[:,2] /= match_distance_xy

    w = spatial.distance_matrix(coords_real, coords_pred)
    w[w > 1] = 9999999 # NEW
    row_ind, col_ind = optimize.linear_sum_assignment(w)

    if return_pairs_coords:
        # multiply coords by match distances
        coords_real[:,0] *= match_distance_t
        coords_real[:,1] *= match_distance_xy
        coords_real[:,2] *= match_distance_xy

        coords_pred[:,0] *= match_distance_t
        coords_pred[:,1] *= match_distance_xy
        coords_pred[:,2] *= match_distance_xy

        # true positive pairs:
        paired_real = [coords_real[i].tolist()
                       for i,j in zip(row_ind,col_ind) if w[i,j]<=1]
        paired_pred = [coords_pred[j].tolist()
                       for i,j in zip(row_ind,col_ind) if w[i,j]<=1]

        # false positive (predictions):
        false_positives = sorted(diff(coords_pred, paired_pred))

        # false negative (annotations):
        false_negatives = sorted(diff(coords_real, paired_real))

        return paired_real, paired_pred, false_positives, false_negatives

    else:
        tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
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

def compute_prec_rec(annotations, preds, thresholds,
                     min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T,
                     min_radius=3, ignore_frames=0, ignore_mask=None):
    '''
    annotations: video of sparks segmentation w/ values in {0,1}
    preds: video of sparks preds w/ values in [0,1]
    thresholds : list of thresholds applied to the preds above which events are kept
    min_dist_xy : minimal spatial distance between two distinct events
    min_dist_t : minimal temporal distance between two distinct events
    min_radius : minimal "radius" of a valid event
    ignore_frames : number of frames ignored at beginning and end of movie
    ignore_mask: binary mask indicating where to ignore the values
    returns : list of Metrics tuples corresponding to thresholds and AUC
    '''

    if ignore_frames != 0:
        annotations = empty_marginal_frames(annotations, ignore_frames)
        preds = empty_marginal_frames(preds, ignore_frames)

    # if using an ignore mask, remove predictions inside ignore regions
    if ignore_mask is not None:
        preds = preds * (1 - ignore_mask)

    metrics = {} # list of 'Metrics' tuples: precision, recall, tp, tp_fp, tp_fn
                 # indexed by threshold value

    coords_true = nonmaxima_suppression(annotations, min_dist_xy, min_dist_t)

    # compute prec and rec for every threshold
    prec = []
    rec = []
    for t in thresholds:
        coords_preds = process_spark_prediction(preds,
                                                t_detection=t,
                                                min_dist_xy=min_dist_xy,
                                                min_dist_t=min_dist_t,
                                                min_radius=min_radius)

        prec_rec = Metrics(*correspondences_precision_recall(coords_real=coords_true,
                                                             coords_pred=coords_preds,
                                                             match_distance_t=min_dist_t,
                                                             match_distance_xy=min_dist_xy))

        metrics[t] = prec_rec
        #print("threshold", t)
        prec.append(prec_rec.precision)
        rec.append(prec_rec.recall)


    # compute AUC for this sample
    #print("PREC", prec)
    #print("REC", rec)
    area_under_curve = auc(rec, prec)

    # TODO: adattare altri scripts che usano questa funzione!!!!
    return metrics, area_under_curve


def reduce_metrics_thresholds(results):
    '''
    apply metrics reduction to results corresponding to different thresholds

    results: dict of Metrics object, indexed by video name (?? TODO: Check!!)
    returns: list of dictionaires for every threshold [reduced_metrics, prec, rec, (auc)]
    '''
    # revert nested dictionaires
    results_t = defaultdict(dict)
    for video_id, video_metrics in results.items():
        for t, t_metrics in video_metrics.items():
            results_t[t][video_id] = t_metrics

    reduced_metrics = {}
    prec = {}
    rec = {}

    for t, res in results_t.items():
        # res is a dict of 'Metrics' for all videos
        reduced_res = reduce_metrics(list(res.values()))

        reduced_metrics[t] = reduced_res
        prec[t] = reduced_res.precision
        rec[t] = reduced_res.recall

    # compute area under the curve for reduced metrics
    #print("REC",rec)
    #print("PREC",prec)
    area_under_curve = auc(list(prec.values()), list(rec.values()))
    #print("AREA UNDER CURVE", area_under_curve)

    # TODO: adattare altri scripts che usano questa funzione!!!!
    return reduced_metrics, prec, rec, area_under_curve



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
                            ignore_frames = 0,
                            convex_hull = False):
    '''
    Get binary clean predictions of puffs (remove small preds)

    pred :          network's puffs predictions
    min_radius :    minimal 'radius' of a valid puff
    ignore_frames : set preds in region ignored by loss fct to 0
    convex_hull :   if true remove holes inside puffs
    '''
    # get binary predictions
    pred_boolean = pred > t_detection

    if convex_hull:
        # remove holes inside puffs (compute convex hull)
        pred_boolean = binary_dilation(pred_boolean, iterations=5)
        pred_boolean = binary_erosion(pred_boolean, iterations=5, border_value=1)

    min_size = (2 * min_radius) ** pred.ndim
    small_objs_removed = morphology.remove_small_objects(pred_boolean,
                                                         min_size=min_size)

    # set first and last frames to 0 according to ignore_frames
    if ignore_frames != 0:
        pred_puffs = empty_marginal_frames(small_objs_removed, ignore_frames)

    return pred_puffs


def process_wave_prediction(pred, t_detection = 0.5,
                            min_radius = 4,
                            ignore_frames = 0):

    # for now: do the same as with puffs

    return process_puff_prediction(pred, t_detection, min_radius, ignore_frames)


def jaccard_score_exclusion_zone(ys,preds,exclusion_radius,
                                 ignore_mask=None,sparks=False):
    '''
    compute IoU score adding exclusion zone if necessary
    ys, preds and ignore_mask are binary masks
    '''

    # Compute intersection and union
    intersection = np.logical_and(ys, preds)
    union = np.logical_or(ys, preds)

    if exclusion_radius != 0:
        # Compute exclusion zone: 1 where Jaccard index has to be computed, 0 otherwise
        dilated = binary_dilation(ys, iterations=exclusion_radius)

        if not sparks:
            eroded = binary_erosion(ys, iterations=exclusion_radius)
            exclusion_mask = 1 - np.logical_xor(eroded,dilated)
        else:
            # Erosion is not computed for spark class
            exclusion_mask = 1 - np.logical_xor(ys,dilated)

        # If ignore mask is given, don't compute values where it is 1
        if ignore_mask is not None:
            # Compute dilation for ignore mask too (erosion not necessary)
            ignore_mask = binary_dilation(ignore_mask, iterations=exclusion_radius)

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
