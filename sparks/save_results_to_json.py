'''
12.04.2022

Use this script to load saved predictions, annotations and movies and compute
results for given parameters (e.g. detection thresholds):
- tp
- tn (if available)
- fp
- fn

Predictions and training annotations are saved in
`.\trainings_validation\{training_name}`

Movies are saved in `..\data\raw_data_and_processing\original_movies`
Raw annotations are saved in `..\data\raw_data_and_processing\original_masks`

Metrics will be saved in
`.\trainings_validation\{training_name}\per_pixel_results`
and
`.\trainings_validation\{training_name}\peaks_results`

Remark: does not work (yet) for models using temporal reduction!
'''

import numpy as np
import glob
import os
import imageio
from collections import defaultdict
import pprint
import json
import configparser

from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score, f1_score

import pandas as pd
import matplotlib.pyplot as plt

import unet
from metrics_tools import (correspondences_precision_recall,
                           Metrics,
                           empty_marginal_frames,
                           compute_puff_wave_metrics,
                           get_sparks_locations_from_mask,
                           get_argmax_segmented_output,
                           nonmaxima_suppression
                          )
from dataset_tools import (load_annotations,
                           load_predictions,
                           load_movies,
                           get_new_mask
                          )


################################################################################
################################## FUNCTIONS ###################################
################################################################################


# compute pixel-based results, for given binary predictions
def get_binary_preds_pixel_based_results(binary_preds, ys, ignore_mask,
                                         min_radius, exclusion_radius,
                                         sparks=False):
    '''
    For given binary preds and annotations of a class and given params,
    compute number of tp, tn, fp and fn pixels.

    binary_preds:       binary UNet predictions with values in {0,1}
    ys:                 binary annotation mask with values in {0,1}
    ignore_mask:        mask == 1 where pixels have been ignored during training
    min_radius:         list of minimal radius of valid predicted events
    exclusion_radius:   list of exclusion radius for metrics computation
    sparks:             if True, do not compute erosion on annotations

    returns:    dict with keys
                min radius x exclusion radius
    '''

    results_dict = defaultdict(dict)

    for min_r in min_radius:
        if min_r > 0:
            # remove small predicted events
            min_size = (2 * min_r) ** binary_preds.ndim
            binary_preds = remove_small_objects(binary_preds, min_size=min_size)

        for exclusion_r in exclusion_radius:
            # compute results wrt to exclusion radius
            tp,tn,fp,fn = compute_puff_wave_metrics(ys=ys,
                                                    preds=binary_preds,
                                                    exclusion_radius=exclusion_r,
                                                    ignore_mask=ignore_mask,
                                                    sparks=sparks,
                                                    results_only=True)

            results_dict[min_r][exclusion_r] = {'tp': tp,
                                                'tn': tn,
                                                'fp': fp,
                                                'fn': fn}

    return results_dict


# compute pixel-based results, using a detection threshold
def get_class_pixel_based_results(raw_preds, ys, ignore_mask,
                                  t_detection, min_radius, exclusion_radius,
                                  sparks=False):
    '''
    For given preds and annotations of a class and given params, compute number
    of tp, tn, fp and fn pixels.

    raw_preds:          raw UNet predictions with values in [0,1]
    ys:                 binary annotation mask with values in {0,1}
    ignore_mask:        mask == 1 where pixels have been ignored during training
    t_detection:        list of detection thresholds
    min_radius:         list of minimal radius of valid predicted events
    exclusion_radius:   list of exclusion radius for metrics computation

    returns:    dict with keys
                t x min radius x exclusion radius
    '''
    results_dict = {}

    for t in t_detection:
        # get binary preds
        binary_preds = raw_preds > t

        results_dict[t] = get_binary_preds_pixel_based_results(binary_preds,
                                                               ys, ignore_mask,
                                                               min_radius,
                                                               exclusion_radius,
                                                               sparks)

    return results_dict


# compute spark peaks results, for given binary prediction
def get_binary_preds_spark_peaks_results(binary_preds, coords_true, movie,
                                         ignore_mask, ignore_frames,
                                         min_radius_sparks):
    '''
    For given binary preds, annotated sparks locations and given params,
    compute number of tp, tp_fp (# preds), tp_fn (# annot) events.

    binary_preds:       raw UNet predictions with values in {0,1}
    coords_true:        list of annotated events
    movie:              sample movie
    ignore_mask:        mask == 1 where pixels have been ignored during training
    ignore_frames:      first and last frames ignored during training
    min_radius_sparks:  list of minimal radius of valid predicted events

    returns:    dict with keys
                min radius
    '''
    results_dict = {}

    for min_r in min_radius_sparks:
        # remove small objects and get clean binary preds
        if min_r > 0:
            min_size = (2 * min_r) ** binary_preds.ndim
            binary_preds = remove_small_objects(binary_preds, min_size=min_size)

        # detect predicted peaks
        coords_pred = nonmaxima_suppression(img=movie,
                                            maxima_mask=binary_preds,
                                            min_dist_xy=MIN_DIST_XY,
                                            min_dist_t=MIN_DIST_T,
                                            return_mask=False,
                                            threshold=0,
                                            sigma=2)

        # remove events in ignored regions
        # in ignored first and last frames...
        if ignore_frames > 0:
            mask_duration = binary_preds.shape[0]
            ignore_frames_up = mask_duration - ignore_frames
            coords_pred = [list(loc) for loc in coords_pred if loc[0]>=ignore_frames and loc[0]<ignore_frames_up]

        # and in ignored mask...
        ignored_pixel_list = np.argwhere(ignore_mask)
        ignored_pixel_list = [list(loc) for loc in ignored_pixel_list]
        coords_pred = [loc for loc in coords_pred if loc not in ignored_pixel_list]

        # compute results (tp, tp_fp, tp_fn)
        results_dict[min_r] = correspondences_precision_recall(coords_real=coords_true,
                                                              coords_pred=coords_pred,
                                                              match_distance_t = MIN_DIST_T,
                                                              match_distance_xy = MIN_DIST_XY,
                                                              return_nb_results = True)

    return results_dict


# compute spark peaks results, using a detection threshold
def get_spark_peaks_results(raw_preds, coords_true, movie, ignore_mask,
                            ignore_frames, t_detection, min_radius_sparks):
        '''
        For given raw preds, annotated sparks locations and given params,
        compute number of tp, tp_fp (# preds), tp_fn (# annot) events.

        raw_preds:          raw UNet predictions with values in [0,1]
        coords_true:        list of annotated events
        movie:              sample movie
        ignore_mask:        mask == 1 where pixels have been ignored during training
        ignore_frames:      first and last frames ignored during training
        t_detection:        list of detection thresholds
        min_radius_sparks:  list of minimal radius of valid predicted events

        returns:    dict with keys
                    t x min radius
        '''
        result_dict = {}

        for t in t_detection:
            # get binary preds
            binary_preds = raw_preds > t

            result_dict[t] = get_binary_preds_spark_peaks_results(binary_preds,
                                                                  coords_true,
                                                                  movie,
                                                                  ignore_mask,
                                                                  ignore_frames,
                                                                  min_radius_sparks)

        return result_dict


# compute average over movies of pixel-based results
def get_sum_results(per_movie_results, pixel_based=True):
    '''
    Given a dict containing the results for all video wrt to all parameters
    (detection threshold/argmax; min radius; (exclusion radius)) return a dict
    containing the sum over all movies, necessary for reducing the metrics.

    per_movie_results:  dict with keys
                        movie_name x t/argmax x min radius (x exclusion radius)
    pixel_based:        if True consider exclusion radius as well

    return:             dict with keys
                        t/argmax x min radius (x exclusion radius)
    '''

    sum_res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # sum values of each movie for every paramenter
    for movie_name, movie_results in per_movie_results.items():
        for t, t_res in movie_results.items():
            for min_r, min_r_res in t_res.items():
                if pixel_based:
                    for excl_r, excl_r_res in min_r_res.items():
                        for res, val in excl_r_res.items():
                            if res in sum_res[t][min_r][excl_r]:
                                sum_res[t][min_r][excl_r][res] += val
                            else:
                                sum_res[t][min_r][excl_r][res] = val
                else:
                    for res, val in min_r_res.items():
                        if res in sum_res[t][min_r]:
                            sum_res[t][min_r][res] += val
                        else:
                            sum_res[t][min_r][res] = val

    return sum_res



if __name__ == "__main__":
    ############################################################################
    ############################# GENERAL SETTINGS #############################
    ############################################################################

    # Select predictions to load
    training_names = ['raw_sparks_lovasz_physio',
                      #"peak_sparks_lovasz_physio",
                      #"peak_sparks_sum_losses_physio"
                     ]

    # Select corresponding config file
    config_files = ['config_raw_sparks_lovasz_physio.ini',
                    #"config_peak_sparks_lovasz_physio.ini",
                    #"config_peak_sparks_sum_losses_physio.ini"
                   ]

    # set simple_mode to True to compute metrics for fewer parameters & thresholds
    simple_mode = True

    # set poster to True if computing the results just for using them in the
    # BDSD22 poster (few params, only sparks/puffs/waves classes, no pixel based
    # metrics for sparks)
    poster = True

    # Load training or testing dataset
    use_train_data = False
    if use_train_data:
        print("Get results for training data")
    else:
        print("Get results for testing data")

    # Set folder where data is loaded and saved
    if not use_train_data:
        metrics_folder = "trainings_validation"
    else :
        metrics_folder = os.path.join("trainings_validation", "train_samples")
    os.makedirs(metrics_folder, exist_ok=True)

    # Set config files folder
    config_folder = "config_files"


    ############################################################################
    ################## LOAD DATA SHARED FOR ALL TRAININGS ######################
    ############################################################################


    # Load raw annotations (sparks unprocessed, most recent version, train and test samples)
    raw_ys_path = os.path.join("..","data","raw_data_and_processing","original_masks")
    raw_ys = load_annotations(raw_ys_path, mask_names="mask")

    # Load original movies (train and test samples)
    movies_path = os.path.join("..","data","raw_data_and_processing","original_movies")
    movies = load_movies(movies_path)


    ############################################################################
    ####################### SET PARAMETERS & THRESHOLDS ########################
    ############################################################################

    # classes for which results will be computed
    classes_list = ['sparks', 'puffs', 'waves', 'sparks_puffs', 'puffs_waves', 'all']
    if poster:
        classes_list = ['sparks', 'puffs', 'waves']

    if simple_mode:
        t_detection = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        min_radius = [0,1,2,3,4,5]
        min_radius_sparks = [0,1]
        exclusion_radius = [0,1,2,3,4,5]
    else:
        t_detection = np.round(np.linspace(0,1,21),2)
        min_radius = [0,1,2,3,4,5,6,7,8,9,10]
        min_radius_sparks = [0,1,2]
        exclusion_radius = [0,1,2,3,4,5,6,7,8,9,10]

    if poster:
        t_detection = np.round(np.linspace(0,1,21),2)
        min_radius = [0,2,4,6]
        min_radius_sparks = [0,1]
        exclusion_radius = [0]

    print("Using parameters:")
    print(f"Detection thresholds: {t_detection}")
    print(f"Puffs and waves' minimal radius: {min_radius}")
    print(f"Sparks' minimal radius: {min_radius_sparks}")
    print(f"Exclusion radius: {exclusion_radius}")

    # physiological params (for spark peaks results)
    PIXEL_SIZE = 0.2 # 1 pixel = 0.2 um x 0.2 um
    MIN_DIST_XY = round(1.8 / PIXEL_SIZE) # min distance in space between sparks
    TIME_FRAME = 6.8 # 1 frame = 6.8 ms
    MIN_DIST_T = round(20 / TIME_FRAME) # min distance in time between sparks


    ############################################################################
    ####################### PROCESS EACH TRAINING MODEL ########################
    ############################################################################





    for training_name, config_name in zip(training_names, config_files):
        print(f"Loading {training_name} UNet predictions...")

        ########################### open config file ###########################

        config_file = os.path.join(config_folder, config_name)
        c = configparser.ConfigParser()
        if os.path.isfile(config_file):
            print(f"Loading {config_file}")
            c.read(config_file)
        else:
            print(f"No config file found at {config_file}.")
            exit()

        epoch = c.getint("testing", "load_epoch")

        ########################## general parameters ##########################

        ignore_frames = c.getint("data", "ignore_frames_loss")

        ################### import .tif files as numpy array ###################

        data_folder = os.path.join(metrics_folder, training_name)
        ys, sparks, puffs, waves = load_predictions(training_name,
                                                    epoch,
                                                    data_folder)

        movie_names = ys.keys()
        # compute results for each sample movie

        pixel_based_results = defaultdict(dict)
        spark_peaks_results = defaultdict(dict)

        for movie_name in movie_names:
            print(f"\tProcessing movie {movie_name}...")

            # get raw predictions for all classes
            # remark:   first and last ignored frames are removed only for
            #           pixel-based results, for spark peaks the events in the
            #           ignored frames are removed after peaks detection
            preds_sample = {'sparks': sparks[movie_name],
                            'puffs': puffs[movie_name],
                            'waves':waves[movie_name]}

            # get raw annotations
            raw_ys_sample = raw_ys[movie_name]
            ys_sample = {'sparks': np.where(raw_ys_sample==1,1,0),
                         'puffs': np.where(raw_ys_sample==3,1,0),
                         'waves': np.where(raw_ys_sample==2,1,0)}

            # get ignore mask (events labelled with 4)
            # remark:   ignore frames can be already removed
            ignore_mask = empty_marginal_frames(np.where(raw_ys_sample==4,1,0),
                                                ignore_frames)

            # get background prediction
            preds_sample['background'] = 1 - preds_sample['sparks'] - preds_sample['puffs'] - preds_sample['waves']
            # get preds as list [background, sparks, waves, puffs]
            preds_sample_list = [preds_sample['background'],
                                 preds_sample['sparks'],
                                 preds_sample['waves'],
                                 preds_sample['puffs']]

            # get argmax binary preds
            argmax_preds_sample, _ = get_argmax_segmented_output(preds=preds_sample_list,
                                                                 get_classes=True)

            for event_class in classes_list:

                if not ((event_class == 'sparks') and poster):
                    ################################################################
                    ########## COMPUTE PIXEL-BASED RESULTS (per movie) #############
                    ################################################################

                    print(f"\tComputing pixel-based results for movie {movie_name} and {event_class} class...")

                    '''
                    Metrics that can be computed using tp, tf, fp, fn:
                    - Jaccard index (IoU)
                    - Dice score
                    - Precision & recall
                    - F-score (e.g. beta = 0.5,1,2)
                    - Accuracy (biased since background is predominant)
                    - Matthews correlation coefficient (MCC)
                    '''

                    '''
                    Class of events that are considered:
                    - sparks
                    - puffs
                    - waves
                    - sparks + puffs
                    - puffs + waves
                    - sparks + puffs + waves (all)
                    '''

                    class_results = {}

                    ######### compute results using a detection threshold ##########

                    if (event_class == 'sparks') and (c.get("data","sparks_type") == 'peaks'):
                        print("WARNING: pixel-based results for sparks when training using peaks are not really meaningful...")

                    # get raw preds and ys
                    if (event_class == 'sparks') or (event_class == 'puffs') or (event_class == 'waves'):
                        class_sample = preds_sample[event_class]
                        ys_class_sample = ys_sample[event_class]
                    elif event_class == 'sparks_puffs':
                        class_sample = preds_sample['sparks']+preds_sample['puffs']
                        ys_class_sample = ys_sample['sparks']+ys_sample['puffs']
                    elif event_class == 'puffs_waves':
                        class_sample = preds_sample['waves']+preds_sample['puffs']
                        ys_class_sample = ys_sample['waves']+ys_sample['puffs']
                    elif event_class == 'all':
                        class_sample = preds_sample['sparks']+preds_sample['puffs']+preds_sample['waves']
                        ys_class_sample = ys_sample['sparks']+ys_sample['puffs']+ys_sample['waves']
                    else:
                        print("WARNING: something is wrong...")

                    # Remarks:  can remove ignored frames, since computing results for
                    #           pixel-based metrics
                    class_sample = empty_marginal_frames(class_sample, ignore_frames)
                    ys_class_sample = empty_marginal_frames(ys_class_sample, ignore_frames)

                    class_results = get_class_pixel_based_results(raw_preds=class_sample,
                                                                  ys=ys_class_sample,
                                                                  ignore_mask=ignore_mask,
                                                                  t_detection=t_detection,
                                                                  min_radius=min_radius_sparks if event_class=='sparks' else min_radius,
                                                                  exclusion_radius=exclusion_radius,
                                                                  sparks=(event_class=='sparks'))

                    ###### compute metrics using argmax values on predictions ######

                    if (event_class == 'sparks') or (event_class == 'puffs') or (event_class == 'waves'):
                        binary_preds_sample = argmax_preds_sample[event_class]
                        ys_class_sample = ys_sample[event_class]
                    elif event_class == 'sparks_puffs':
                        binary_preds_sample = argmax_preds_sample['sparks']+argmax_preds_sample['puffs']
                        ys_class_sample = ys_sample['sparks']+ys_sample['puffs']
                    elif event_class == 'puffs_waves':
                        binary_preds_sample = argmax_preds_sample['waves']+argmax_preds_sample['puffs']
                        ys_class_sample = ys_sample['waves']+ys_sample['puffs']
                    elif event_class == 'all':
                        binary_preds_sample = argmax_preds_sample['sparks']+argmax_preds_sample['puffs']+argmax_preds_sample['waves']
                        ys_class_sample = ys_sample['sparks']+ys_sample['puffs']+ys_sample['waves']
                    else:
                        print("WARNING: something is wrong...")

                    # Get binary preds as boolean array
                    binary_preds_sample = np.array(binary_preds_sample, dtype=bool)

                    # Remarks:  can remove ignored frames, since computing results for
                    #           pixel-based metrics
                    binary_preds_sample = empty_marginal_frames(binary_preds_sample, ignore_frames)
                    ys_class_sample = empty_marginal_frames(ys_class_sample, ignore_frames)

                    class_results['argmax'] = get_binary_preds_pixel_based_results(binary_preds=binary_preds_sample,
                                                                                   ys=ys_class_sample,
                                                                                   ignore_mask=ignore_mask,
                                                                                   min_radius=min_radius_sparks if event_class=='sparks' else min_radius,
                                                                                   exclusion_radius=exclusion_radius,
                                                                                   sparks=(event_class=='sparks'))

                    # store class results in dict
                    pixel_based_results[event_class][movie_name] = class_results

                ################################################################
                ################# COMPUTE SPARK PEAKS RESULTS ##################
                ################################################################

                if event_class == 'sparks':

                    print(f"\tComputing spark peaks results for movie {movie_name}...")

                    '''
                    Metrics that can be computed using tp, tp_fp, tp_fn:
                    - Precision & recall
                    - F-score (e.g. beta = 0.5,1,2)
                    (- Matthews correlation coefficient (MCC))???
                    '''

                    '''
                    Class of events that are considered:
                    - sparks
                    '''

                    sparks_results = {}

                    # get sample movie
                    movie_sample = movies[movie_name]

                    # get ys used during training
                    ys_sample_training = ys[movie_name]

                    # extract peak locations from annotations used during training
                    if c.get("data","sparks_type") == 'peaks':
                        coords_true = get_sparks_locations_from_mask(mask=ys_sample_training,
                                                                     min_dist_xy=MIN_DIST_XY,
                                                                     min_dist_t=MIN_DIST_T,
                                                                     ignore_frames=ignore_frames)
                    elif c.get("data","sparks_type") == 'raw':
                        print("\t\tModel trained using raw sparks, extracting locations from annotations...")
                        coords_true = get_new_mask(video=movie_sample,
                                                   mask=ys_sample_training,
                                                   min_dist_xy=MIN_DIST_XY,
                                                   min_dist_t=MIN_DIST_T,
                                                   return_loc=True)
                    else:
                        print("WARNING: something is wrong...")

                    # remove events ignored by loss function
                    if ignore_frames > 0:
                        if len(coords_true) > 0:
                            mask_duration = ys_sample_training.shape[0]
                            ignore_frames_up = mask_duration - ignore_frames
                            coords_true = [loc for loc in coords_true
                                           if loc[0]>=ignore_frames and loc[0]<ignore_frames_up]

                    ####### compute results using a detection threshold ########

                    sparks_results = get_spark_peaks_results(raw_preds=preds_sample[event_class],
                                                             coords_true=coords_true,
                                                             movie=movie_sample,
                                                             ignore_mask=ignore_mask,
                                                             ignore_frames=ignore_frames,
                                                             t_detection=t_detection,
                                                             min_radius_sparks=min_radius_sparks)

                    #### compute metrics using argmax values on predictions ####

                    binary_preds_sample = argmax_preds_sample[event_class]
                    binary_preds_sample = np.array(binary_preds_sample, dtype=bool)

                    sparks_results['argmax'] = get_binary_preds_spark_peaks_results(binary_preds=binary_preds_sample,
                                                                                    coords_true=coords_true,
                                                                                    movie=movie_sample,
                                                                                    ignore_mask=ignore_mask,
                                                                                    ignore_frames=ignore_frames,
                                                                                    min_radius_sparks=min_radius_sparks)

                    # store class results in dict
                    spark_peaks_results[event_class][movie_name] = sparks_results

        ########################################################################
        ####### COMPUTE RESULTS AVERAGED ON ALL MOVIES AND SAVE TO DISK ########
        ########################################################################

        for event_class in classes_list:
            ####################### pixel-based metrics ########################
            pixel_based_results[event_class]['average'] = get_sum_results(pixel_based_results[event_class])

            # save results dict on disk as json
            data_folder = os.path.join(metrics_folder, training_name,
                                       "per_pixel_results")
            os.makedirs(data_folder, exist_ok=True)

            with open(os.path.join(data_folder, event_class+"_results.json"),"w") as f:
                json.dump(pixel_based_results[event_class],f)

            ####################### event-based metrics ########################
            if event_class == 'sparks':
                spark_peaks_results[event_class]['average'] =  get_sum_results(spark_peaks_results[event_class],
                                                                               pixel_based=False)

                # save results dict on disk as json
                data_folder = os.path.join(metrics_folder, training_name,
                                           "spark_peaks_results")
                os.makedirs(data_folder, exist_ok=True)

                with open(os.path.join(data_folder, event_class+"_results.json"),"w") as f:
                    json.dump(spark_peaks_results[event_class],f)
