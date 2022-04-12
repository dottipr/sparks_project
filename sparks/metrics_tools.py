'''
This script will contain methods useful for processing the unet outputs
'''
import glob
import imageio
import os
import matplotlib.pyplot as plt

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
           "correspondences_precision_recall",
           "reduce_metrics",
           "empty_marginal_frames",
           "write_videos_on_disk",
           "compute_prec_rec",
           "reduce_metrics_thresholds",
           "compute_f1_score",
           "take_closest",
           "get_sparks_locations_from_mask",
           "process_spark_prediction",
           "process_puff_prediction",
           "process_wave_prediction",
           "compute_puff_wave_metrics",
           "compute_average_puff_wave_metrics",
           "get_argmax_segmented_output"
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


def get_argmax_segmented_output(preds, get_classes=True):
    '''
    preds are the (exponential) raw outputs of the unet for each class:
    [background, sparks, waves, puffs] (4 x duration x 64 x 512)
    '''
    argmax_classes = np.argmax(preds, axis=0)

    if not get_classes:
        return argmax_classes

    preds = {}
    preds['sparks'] = np.where(argmax_classes==1,1,0)
    preds['waves'] = np.where(argmax_classes==2,1,0)
    preds['puffs'] = np.where(argmax_classes==3,1,0)

    return preds, argmax_classes




################################ Sparks metrics ################################

'''
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
'''

Metrics = namedtuple('Metrics', ['precision', 'recall', 'f1_score', 'tp', 'tp_fp', 'tp_fn'])


#def in_bounds(points, shape):
#
#    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
#                                for coords_i, shape_i in zip(points.T, shape)])

def filter_nan_gaussian_david(arr, sigma):
    """Allows intensity to leak into the nan area.
    According to Davids answer:
        https://stackoverflow.com/a/36307291/7128154
    """
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = ndi.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)

    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = ndi.gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = np.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[np.isnan(arr)] = np.nan
    return gauss

def nonmaxima_suppression(img,maxima_mask=None,
                          min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T,
                          return_mask=False, threshold=0.5, sigma=2):
    '''
    Extract local maxima from input array (t,x,y).
    img :           input array
    maxima_mask :   if not None, look for local maxima only inside the mask
    min_dist_xy :   minimal spatial distance between two maxima
    min_dist_t :    minimal temporal distance between two maxima
    return_mask :   if True return both masks with maxima and locations, if
                    False only returns locations
    threshold :     minimal value of maximum points
    sigma :         sigma parameter of gaussian filter
    '''
    img = img.astype(np.float)

    ''' too many peaks !!!!
    if maxima_mask is not None:
        # https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        # apply dilation to maxima mask
        #maxima_mask_dilated = ndi.binary_dilation(maxima_mask, iterations=sigma) # "W"
        maxima_mask_dilated = maxima_mask
        # mask out region from img with dilated mask
        masked_img = np.where(maxima_mask_dilated, img, np.nan) # "V"
        imageio.volwrite("TEST_masked_video.tif", masked_img)

        # smooth masked input image
        smooth_img = filter_nan_gaussian_david(masked_img, sigma)
        smooth_img[np.isnan(smooth_img)] = 0.
        #smooth_masked_img = ndi.gaussian_filter(masked_img, sigma=1) # "VV"
        #smooth_mask = ndi.gaussian_filter(maxima_mask_dilated.astype(np.float), sigma=1) # "WW"
        #imageio.volwrite("TEST_smooth_masked_video.tif", smooth_masked_img)
        #imageio.volwrite("TEST_smooth_mask.tif", smooth_mask)

        #smooth_img = smooth_masked_img/smooth_mask # "VV/WW"
        #smooth_img[masked_img==np.nan] = 0
        imageio.volwrite("TEST_smooth_video.tif", smooth_img)
        print(np.unique(smooth_img))

        # DEBUG
        original_smoothed = ndi.gaussian_filter(img, sigma=sigma)
        original_smoothed_masked = np.where(maxima_mask_dilated, original_smoothed, 0.)
        difference = np.where(original_smoothed_masked != smooth_img, 1., 0.)
        imageio.volwrite("TEST_DEBUG.tif", difference)



    else:
        smooth_img = ndi.gaussian_filter(img, sigma=sigma)

    # search for local maxima
    min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
    dilated = ndi.maximum_filter(smooth_img,
                                 footprint=min_dist)
    imageio.volwrite("TEST_dilated.tif", dilated)
    argmaxima = np.logical_and(smooth_img == dilated, smooth_img > threshold)
    imageio.volwrite("TEST_maxima.tif", np.uint8(argmaxima))
    #imageio.volwrite("TEST_all_video_maxima.tif", np.uint8(np.logical_and(smooth_img == dilated, smooth_img > threshold)))'''

    # multiply values of video inside maxima mask
    #img = np.where(maxima_mask, img*1.5, img)
    imageio.volwrite("TEST_DEBUG.tif", img)

    smooth_img = ndi.gaussian_filter(img, sigma=sigma)
    imageio.volwrite("TEST_smooth_video.tif", smooth_img)

    if maxima_mask is not None:
        # apply dilation to mask
        #maxima_mask_dilated = ndi.binary_dilation(maxima_mask, iterations=sigma)
        maxima_mask_dilated = maxima_mask
        # set pixels outside maxima_mask to zero
        masked_img = np.where(maxima_mask_dilated, smooth_img, 0.)
        #masked_img = np.where(maxima_mask, smooth_img, 0.)
        imageio.volwrite("TEST_masked_video.tif", masked_img)
    else:
        masked_img = smooth_img


    # compute shape for maximum filter
    #min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
    radius = round(min_dist_xy/2)
    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
    disk = x**2+y**2 <= radius**2
    min_dist = np.stack([disk]*min_dist_t, axis=0)

    # detect local maxima
    dilated = ndi.maximum_filter(smooth_img,
                                 footprint=min_dist)
    imageio.volwrite("TEST_dilated.tif", dilated)
    argmaxima = np.logical_and(smooth_img == dilated, masked_img > threshold)
    imageio.volwrite("TEST_maxima.tif", np.uint8(argmaxima))
    #imageio.volwrite("TEST_all_video_maxima.tif", np.uint8(np.logical_and(smooth_img == dilated, smooth_img > threshold)))

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
    #sparks_mask = empty_marginal_frames(sparks_mask, ignore_frames)
    coords = nonmaxima_suppression(img=sparks_mask,
                                   min_dist_xy=min_dist_xy,
                                   min_dist_t=min_dist_t,
                                   threshold=0, sigma=1)

    # remove first and last frames
    if ignore_frames > 0:
        mask_duration = mask.shape[0]
        ignore_frames_up = mask_duration - ignore_frames
        coords = [loc for loc in coords if loc[0]>=ignore_frames and loc[0]<ignore_frames_up]

    return coords


def process_spark_prediction(pred,
                             movie=None,
                             t_detection = 0.9,
                             min_dist_xy = MIN_DIST_XY,
                             min_dist_t = MIN_DIST_T,
                             min_radius = 3,
                             return_mask = False,
                             return_clean_pred = False,
                             ignore_frames = 0,
                             sigma = 2):
    '''
    Get sparks centres from preds: remove small events + nonmaxima suppression

    pred: network's sparks predictions
    movie: original sample movie
    t_detection: sparks detection threshold
    min_dist_xy : minimal spatial distance between two maxima
    min_dist_t : minimal temporal distance between two maxima
    min_radius: minimal 'radius' of a valid spark
    return_mask: if True return mask and locations of sparks
    return_clean_pred: if True only return preds without small events
    ignore_frames: set preds in region ignored by loss fct to 0
    sigma: sigma value used in gaussian smoothing in nonmaxima suppression
    '''
    # get binary preds
    pred_boolean = pred > t_detection

    # remove small objects and get clean binary preds
    if min_radius > 0:
        min_size = (2 * min_radius) ** pred.ndim
        small_objs_removed = morphology.remove_small_objects(pred_boolean,
                                                             min_size=min_size)
    else:
        small_objs_removed = pred_boolean


    # remove first and last object from sparks mask
    #small_objs_removed = empty_marginal_frames(small_objs_removed,
    #                                           ignore_frames)

    #imageio.volwrite("TEST_small_objs_removed.tif", np.uint8(small_objs_removed))
    #imageio.volwrite("TEST_clean_preds.tif", np.where(small_objs_removed, pred, 0))
    if return_clean_pred:
        # original movie without small objects:
        big_pred = np.where(small_objs_removed, pred, 0)
        return big_pred

    assert movie is not None, "Provide original movie to detect spark peaks"

    # detect events (nonmaxima suppression)
    argwhere, argmaxima = nonmaxima_suppression(img=movie,
                                                maxima_mask=small_objs_removed,
                                                min_dist_xy=min_dist_xy,
                                                min_dist_t=min_dist_t,
                                                return_mask=True,
                                                threshold=0,
                                                sigma=sigma)

    # remove first and last frames
    if ignore_frames > 0:
        mask_duration = pred.shape[0]
        ignore_frames_up = mask_duration - ignore_frames
        argwhere = [loc for loc in argwhere if loc[0]>=ignore_frames and loc[0]<ignore_frames_up]

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
    if coords_real.size > 0:
        coords_real[:,0] /= match_distance_t
        coords_real[:,1] /= match_distance_xy
        coords_real[:,2] /= match_distance_xy

    if coords_pred.size > 0:
        coords_pred[:,0] /= match_distance_t
        coords_pred[:,1] /= match_distance_xy
        coords_pred[:,2] /= match_distance_xy # check if integer!!!!!!!!!!!

    if coords_real.size and coords_pred.size > 0:
        w = spatial.distance_matrix(coords_real, coords_pred)
        w[w > 1] = 9999999 # NEW
        row_ind, col_ind = optimize.linear_sum_assignment(w)

    if return_pairs_coords:
        if coords_real.size > 0:
            # multiply coords by match distances
            coords_real[:,0] *= match_distance_t
            coords_real[:,1] *= match_distance_xy
            coords_real[:,2] *= match_distance_xy

        if coords_pred.size > 0:
            coords_pred[:,0] *= match_distance_t
            coords_pred[:,1] *= match_distance_xy
            coords_pred[:,2] *= match_distance_xy

        if coords_real.size and coords_pred.size > 0:
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
            return [], [], coords_pred, coords_real

    else:
        if coords_real.size and coords_pred.size > 0:
            tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
        else:
            tp = 0

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

        f1_score = compute_f1_score(precision,recall)

        return precision, recall, f1_score, tp, tp_fp, tp_fn


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

    f1_score = compute_f1_score(precision,recall)

    return Metrics(precision, recall, f1_score, tp, tp_fp, tp_fn)

def compute_prec_rec(annotations, preds, movie, thresholds,
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

    metrics = {} # list of 'Metrics' tuples: precision, recall, f1_score, tp, tp_fp, tp_fn
                 # indexed by threshold value

    coords_true = get_sparks_locations_from_mask(annotations,
                                                 min_dist_xy,
                                                 min_dist_t)

    # compute prec and rec for every threshold
    prec = []
    rec = []
    for t in thresholds:
        coords_preds = process_spark_prediction(preds,
                                                movie,
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
    #area_under_curve = auc(rec, prec)

    # TODO: adattare altri scripts che usano questa funzione!!!!
    return metrics#, area_under_curve


def reduce_metrics_thresholds(results):
    '''
    apply metrics reduction to results corresponding to different thresholds

    results: dict of Metrics object, indexed by video name (?? TODO: Check!!)
    returns: list of dictionaires for every threshold [reduced_metrics, prec, rec, f1_score]
    '''
    # revert nested dictionaires
    results_t = defaultdict(dict)
    for video_id, video_metrics in results.items():
        for t, t_metrics in video_metrics.items():
            results_t[t][video_id] = t_metrics

    reduced_metrics = {}
    prec = {}
    rec = {}
    f1_score = {}

    for t, res in results_t.items():
        # res is a dict of 'Metrics' for all videos
        reduced_res = reduce_metrics(list(res.values()))

        reduced_metrics[t] = reduced_res
        prec[t] = reduced_res.precision
        rec[t] = reduced_res.recall
        f1_score[t] = reduced_res.f1_score


    # compute area under the curve for reduced metrics
    #print("REC",rec)
    #print("PREC",prec)
    #area_under_curve = auc(list(prec.values()), list(rec.values()))
    #print("AREA UNDER CURVE", area_under_curve)

    # TODO: adattare altri scripts che usano questa funzione!!!!
    return reduced_metrics, prec, rec, f1_score#, area_under_curve


def compute_f1_score(prec,rec):
    f1_score = 2*prec*rec/(prec+rec) if prec+rec != 0 else 0.
    return f1_score

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


def compute_puff_wave_metrics(ys, preds, exclusion_radius,
                              ignore_mask=None, sparks=False):
    '''
    Compute some metrics given labels and predicted segmentation.
    ys : annotated segmentation
    preds : predicted segmentation
    exclusion_radius : radius around ys border which is ignored for metrics
    ignore_mask : mask that is ignored by loss function during training
    sparks : if true, do not compute erosion on sparks for exclusion_radius
    '''

    tp = np.logical_and(ys, preds)
    tn = np.logical_not(np.logical_or(ys, preds))
    fp = np.logical_and(preds, np.logical_not(tp))
    fn = np.logical_and(ys, np.logical_not(tp))

    assert (np.sum(tp)+np.sum(tn)+np.sum(fp)+np.sum(fn) == ys.size)

    # compute exclusion zone if required
    if exclusion_radius > 0:
        dilated = binary_dilation(ys, iterations=exclusion_radius)

        if not sparks:
            eroded = binary_erosion(ys, iterations=exclusion_radius)
            exclusion_mask = np.logical_not(np.logical_xor(eroded,dilated))
        else:
            # do not erode sparks
            exclusion_mask = np.logical_not(np.logical_xor(ys,dilated))

        tp = np.logical_and(tp, exclusion_mask)
        tn = np.logical_and(tn, exclusion_mask)
        fp = np.logical_and(fp, exclusion_mask)
        fn = np.logical_and(fn, exclusion_mask)

    # define mask where pixels aren't ignored by loss function
    if ignore_mask is not None:

        if exclusion_radius > 0:
            # compute dilation for ignore mask (erosion not necessary)
            ignore_mask = binary_dilation(ignore_mask,
                                          iterations=exclusion_radius)

        compute_mask = np.logical_not(ignore_mask)

        tp = np.logical_and(tp, compute_mask)
        tn = np.logical_and(tn, compute_mask)
        fp = np.logical_and(fp, compute_mask)
        fn = np.logical_and(fn, compute_mask)

    # compute all metrics
    n_tp = np.count_nonzero(tp)
    n_tn = np.count_nonzero(tn)
    n_fp = np.count_nonzero(fp)
    n_fn = np.count_nonzero(fn)

    iou = n_tp/(n_tp+n_fn+n_fp) if (n_tp+n_fn+n_fp) != 0 else 1.0
    prec = n_tp/(n_tp+n_fp) if (n_tp+n_fp) != 0 else 1.0
    rec = n_tp/(n_tp+n_fn) if (n_tp+n_fn) != 0 else 1.0
    accuracy = (n_tp+n_tn)/(n_tp+n_tn+n_fp+n_fn)

    return {"iou": iou,
            "prec": prec,
            "rec": rec,
            "accuracy": accuracy}

def compute_average_puff_wave_metrics(metrics):
    '''
    compute average of puff+wave metrics over all movies
    metrics: dict whose indices are movie names and each entry contains the
             metrics {'accuracy', 'iou', 'prec', 'rec'}
    return a dict indexed by {'accuracy', 'iou', 'prec', 'rec'}
    '''
    # reverse dict
    metrics_dict = defaultdict(lambda: defaultdict(dict))
    for movie_name, movie_metrics in metrics.items():
        for metric_name, val in movie_metrics.items():
            metrics_dict[metric_name][movie_name] = val

    res = {metric_name: sum(res.values())/len(res)
            for metric_name, res in metrics_dict.items()}

    return res


''' OLD: adapt code to use compute_puff_wave_metrics instead
def jaccard_score_exclusion_zone(ys,preds,exclusion_radius,
                                 ignore_mask=None,sparks=False):

    #compute IoU score adding exclusion zone if necessary
    #ys, preds and ignore_mask are binary masks


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
            compute_mask = np.logical_and(1 - ignore_mask, exclusion_mask)
    else:
        compute_mask = 1 - ignore_mask

        # Compute intersecion of exclusion mask with intersection and union
        intersection = np.logical_and(intersection, compute_mask)
        union = np.logical_and(union, compute_mask)

    #print("Pixels in intersection:", np.count_nonzero(intersection))
    #print("Pixels in union:", np.count_nonzero(union))

    if np.count_nonzero(union) != 0:
        iou = np.count_nonzero(intersection)/np.count_nonzero(union)
    else:
        iou = 1.

    return iou
'''
