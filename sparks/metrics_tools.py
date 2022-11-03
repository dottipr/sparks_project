'''
24.10.2022

Script with function for metrics on UNet output computation.

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as ..., )
'''
from collections import namedtuple, defaultdict

import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import optimize, spatial
from sklearn.metrics import auc
from bisect import bisect_left

from data_processing_tools import (empty_marginal_frames,
                                   process_spark_prediction,
                                   simple_nonmaxima_suppression,
                                   sparks_connectivity_mask)



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



############################### Generic metrics ################################


def compute_iou(ys_roi, preds_roi, ignore_mask=None):
    '''
    Compute IoU for given single annotated and predicted events.
    ys_roi :            annotated event ROI
    preds_roi :         predicted event ROI
    ignore_mask :       mask that is ignored by loss function during training
    '''
    # define mask where pixels aren't ignored by loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    intersection = np.logical_and(ys_roi, preds_roi_real)
    union = np.logical_or(ys_roi, preds_roi_real)
    iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    return iou


def compute_inter_min(ys_roi, preds_roi, ignore_mask=None):
    '''
    Compute intersection over minimum area for given single annotated and predicted events.
    ys_roi :            annotated event ROI
    preds_roi :         predicted event ROI
    ignore_mask :       mask that is ignored by loss function during training
    '''
    # define mask where pixels aren't ignored by loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    intersection = np.logical_and(ys_roi, preds_roi_real)
    ys_area = np.count_nonzero(ys_roi)
    preds_area = np.count_nonzero(preds_roi_real)

    if preds_area > 0:
        iomin = np.count_nonzero(intersection) / min(preds_area, ys_area)
    else:
        iomin = 0
    return iomin



################################ Sparks metrics ################################

'''
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
'''

Metrics = namedtuple('Metrics', ['precision',
                                 'recall',
                                 'f1_score',
                                 'tp',
                                 'tp_fp',
                                 'tp_fn'])



def correspondences_precision_recall(coords_real, coords_pred,
                                     match_distance_t = MIN_DIST_T,
                                     match_distance_xy = MIN_DIST_XY,
                                     return_pairs_coords = False,
                                     return_nb_results = False):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.

    If return_pairs_coords == True, return paired sparks coordinated
    If return_nb_results == True, return only tp, tp_fp, tp_fn as a dict
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

            if return_nb_results:
                tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {'tp': tp,
                       'tp_fp': tp_fp,
                       'tp_fn': tp_fn}
                return res, paired_real, paired_pred, false_positives, false_negatives
            else:
                return paired_real, paired_pred, false_positives, false_negatives
        else:
            if return_nb_results:
                tp = 0
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {'tp': tp,
                       'tp_fp': tp_fp,
                       'tp_fn': tp_fn}
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
            return {'tp': tp,
                    'tp_fp': tp_fp,
                    'tp_fn': tp_fn}

        if tp_fp > 0:
            precision = tp / tp_fp
        else:
            precision = 1.0

        if tp_fn > 0:
            recall = tp / tp_fn
        else:
            recall = 1.0

        f1_score = compute_f_score(precision,recall)

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

    f1_score = compute_f_score(precision,recall)

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

    connectivity_mask = sparks_connectivity_mask(min_dist_xy, min_dist_t)
    coords_true = simple_nonmaxima_suppression(img=movie,
                                               maxima_mask=annotations,
                                               min_dist=connectivity_mask,
                                               return_mask=False,
                                               threshold=0,
                                               sigma=2)

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


def compute_f_score(prec,rec,beta=1):
    if beta == 1:
        f_score = 2*prec*rec/(prec+rec) if prec+rec != 0 else 0.
    else:
        f_score = (1+beta*beta)*(prec+rec)/(beta*beta*prec+rec) if prec+rec != 0 else 0.
    return f_score

########################## Segmentation-based metrics ###########################

'''
Utils for computing metrics related to puffs and waves, e.g.:
- Jaccard index
- exclusion region for Jaccard index
'''


def compute_puff_wave_metrics(ys, preds, exclusion_radius,
                              ignore_mask=None, sparks=False,
                              results_only=False):
    '''
    Compute some metrics given labels and predicted segmentation.
    ys :                annotated segmentation
    preds :             predicted segmentation
    exclusion_radius :  radius around ys border which is ignored for metrics
    ignore_mask :       mask that is ignored by loss function during training
    sparks :            if True, do not compute erosion on sparks for
                        exclusion_radius
    results_only:       if True, return number of tp, tn, fp, fn pixels
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

    if results_only:
        return n_tp, n_tn, n_fp, n_fn

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




############################### Unused functions ###############################


#def in_bounds(points, shape):
#
#    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
#                                for coords_i, shape_i in zip(points.T, shape)])


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


#def reduce_metrics_thresholds(results):
#    '''
#    apply metrics reduction to results corresponding to different thresholds
#
#    results: dict of Metrics object, indexed by video name (?? TODO: Check!!)
#    returns: list of dictionaires for every threshold [reduced_metrics, prec, rec, f1_score]
#    '''
#    # revert nested dictionaires
#    results_t = defaultdict(dict)
#    for video_id, video_metrics in results.items():
#        for t, t_metrics in video_metrics.items():
#            results_t[t][video_id] = t_metrics
#
#    reduced_metrics = {}
#    prec = {}
#    rec = {}
#    f1_score = {}
#
#    for t, res in results_t.items():
#        # res is a dict of 'Metrics' for all videos
#        reduced_res = reduce_metrics(list(res.values()))
#
#        reduced_metrics[t] = reduced_res
#        prec[t] = reduced_res.precision
#        rec[t] = reduced_res.recall
#        f1_score[t] = reduced_res.f1_score
#
#
#    # compute area under the curve for reduced metrics
#    #print("REC",rec)
#    #print("PREC",prec)
#    #area_under_curve = auc(list(prec.values()), list(rec.values()))
#    #print("AREA UNDER CURVE", area_under_curve)
#
#    # TODO: adattare altri scripts che usano questa funzione!!!!
#    return reduced_metrics, prec, rec, f1_score#, area_under_curve


################# fcts from save_results_to_json.py (old file) #################

# compute pixel-based results, for given binary predictions
#def get_binary_preds_pixel_based_results(binary_preds, ys, ignore_mask,
#                                         min_radius, exclusion_radius,
#                                         sparks=False):
#    '''
#    For given binary preds and annotations of a class and given params,
#    compute number of tp, tn, fp and fn pixels.
#
#    binary_preds:       binary UNet predictions with values in {0,1}
#    ys:                 binary annotation mask with values in {0,1}
#    ignore_mask:        mask == 1 where pixels have been ignored during training
#    min_radius:         list of minimal radius of valid predicted events
#    exclusion_radius:   list of exclusion radius for metrics computation
#    sparks:             if True, do not compute erosion on annotations
#
#    returns:    dict with keys
#                min radius x exclusion radius
#    '''
#
#    results_dict = defaultdict(dict)
#
#    for min_r in min_radius:
#        if min_r > 0:
#            # remove small predicted events
#            min_size = (2 * min_r) ** binary_preds.ndim
#            binary_preds = remove_small_objects(binary_preds, min_size=min_size)
#
#        for exclusion_r in exclusion_radius:
#            # compute results wrt to exclusion radius
#            tp,tn,fp,fn = compute_puff_wave_metrics(ys=ys,
#                                                    preds=binary_preds,
#                                                    exclusion_radius=exclusion_r,
#                                                    ignore_mask=ignore_mask,
#                                                    sparks=sparks,
#                                                    results_only=True)
#
#            results_dict[min_r][exclusion_r] = {'tp': tp,
#                                                'tn': tn,
#                                                'fp': fp,
#                                                'fn': fn}
#
#    return results_dict


# compute pixel-based results, using a detection threshold
#def get_class_pixel_based_results(raw_preds, ys, ignore_mask,
#                                  t_detection, min_radius, exclusion_radius,
#                                  sparks=False):
#    '''
#    For given preds and annotations of a class and given params, compute number
#    of tp, tn, fp and fn pixels.
#
#    raw_preds:          raw UNet predictions with values in [0,1]
#    ys:                 binary annotation mask with values in {0,1}
#    ignore_mask:        mask == 1 where pixels have been ignored during training
#    t_detection:        list of detection thresholds
#    min_radius:         list of minimal radius of valid predicted events
#    exclusion_radius:   list of exclusion radius for metrics computation
#
#    returns:    dict with keys
#                t x min radius x exclusion radius
#    '''
#    results_dict = {}
#
#    for t in t_detection:
#        # get binary preds
#        binary_preds = raw_preds > t
#
#        results_dict[t] = get_binary_preds_pixel_based_results(binary_preds,
#                                                               ys, ignore_mask,
#                                                               min_radius,
#                                                               exclusion_radius,
#                                                               sparks)
#
#    return results_dict


# compute spark peaks results, for given binary prediction
#def get_binary_preds_spark_peaks_results(binary_preds, coords_true, movie,
#                                         ignore_mask, ignore_frames,
#                                         min_radius_sparks):
#    '''
#    For given binary preds, annotated sparks locations and given params,
#    compute number of tp, tp_fp (# preds), tp_fn (# annot) events.
#
#    binary_preds:       raw UNet predictions with values in {0,1}
#    coords_true:        list of annotated events
#    movie:              sample movie
#    ignore_mask:        mask == 1 where pixels have been ignored during training
#    ignore_frames:      first and last frames ignored during training
#    min_radius_sparks:  list of minimal radius of valid predicted events
#
#    returns:    dict with keys
#                min radius
#    '''
#    results_dict = {}
#
#    for min_r in min_radius_sparks:
#        # remove small objects and get clean binary preds
#        if min_r > 0:
#            min_size = (2 * min_r) ** binary_preds.ndim
#            binary_preds = remove_small_objects(binary_preds, min_size=min_size)
#
#        # detect predicted peaks
#        connectivity_mask = sparks_connectivity_mask(MIN_DIST_XY,MIN_DIST_T)
#        coords_pred =simple_nonmaxima_suppression(img=movie,
#                                                  maxima_mask=binary_preds,
#                                                  min_dist=connectivity_mask,
#                                                  return_mask=False,
#                                                  threshold=0,
#                                                  sigma=2)
#
#        # remove events in ignored regions
#        # in ignored first and last frames...
#        if ignore_frames > 0:
#            mask_duration = binary_preds.shape[0]
#            ignore_frames_up = mask_duration - ignore_frames
#            coords_pred = [list(loc) for loc in coords_pred if loc[0]>=ignore_frames and loc[0]<ignore_frames_up]
#
#        # and in ignored mask...
#        ignored_pixel_list = np.argwhere(ignore_mask)
#        ignored_pixel_list = [list(loc) for loc in ignored_pixel_list]
#        coords_pred = [loc for loc in coords_pred if loc not in ignored_pixel_list]
#
#        # compute results (tp, tp_fp, tp_fn)
#        results_dict[min_r] = correspondences_precision_recall(coords_real=coords_true,
#                                                              coords_pred=coords_pred,
#                                                              match_distance_t = MIN_DIST_T,
#                                                              match_distance_xy = MIN_DIST_XY,
#                                                              return_nb_results = True)
#
#    return results_dict


# compute spark peaks results, using a detection threshold
#def get_spark_peaks_results(raw_preds, coords_true, movie, ignore_mask,
#                            ignore_frames, t_detection, min_radius_sparks):
#        '''
#        For given raw preds, annotated sparks locations and given params,
#        compute number of tp, tp_fp (# preds), tp_fn (# annot) events.
#
#        raw_preds:          raw UNet predictions with values in [0,1]
#        coords_true:        list of annotated events
#        movie:              sample movie
#        ignore_mask:        mask == 1 where pixels have been ignored during training
#        ignore_frames:      first and last frames ignored during training
#        t_detection:        list of detection thresholds
#        min_radius_sparks:  list of minimal radius of valid predicted events
#
#        returns:    dict with keys
#                    t x min radius
#        '''
#        result_dict = {}
#
#        for t in t_detection:
#            # get binary preds
#            binary_preds = raw_preds > t
#
#            result_dict[t] = get_binary_preds_spark_peaks_results(binary_preds,
#                                                                  coords_true,
#                                                                  movie,
#                                                                  ignore_mask,
#                                                                  ignore_frames,
#                                                                  min_radius_sparks)
#
#        return result_dict


# compute average over movies of pixel-based results
#def get_sum_results(per_movie_results, pixel_based=True):
#    '''
#    Given a dict containing the results for all video wrt to all parameters
#    (detection threshold/argmax; min radius; (exclusion radius)) return a dict
#    containing the sum over all movies, necessary for reducing the metrics.
#
#    per_movie_results:  dict with keys
#                        movie_name x t/argmax x min radius (x exclusion radius)
#    pixel_based:        if True consider exclusion radius as well
#
#    return:             dict with keys
#                        t/argmax x min radius (x exclusion radius)
#    '''
#
#    sum_res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
#
#    # sum values of each movie for every paramenter
#    for movie_name, movie_results in per_movie_results.items():
#        for t, t_res in movie_results.items():
#            for min_r, min_r_res in t_res.items():
#                if pixel_based:
#                    for excl_r, excl_r_res in min_r_res.items():
#                        for res, val in excl_r_res.items():
#                            if res in sum_res[t][min_r][excl_r]:
#                                sum_res[t][min_r][excl_r][res] += val
#                            else:
#                                sum_res[t][min_r][excl_r][res] = val
#                else:
#                    for res, val in min_r_res.items():
#                        if res in sum_res[t][min_r]:
#                            sum_res[t][min_r][res] += val
#                        else:
#                            sum_res[t][min_r][res] = val
#
#    return sum_res
