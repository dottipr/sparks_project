'''
15.03.2022

Use this script to load saved predictions, annotations and movies and compute
metrics:
- plain IoU for puffs and waves
- IoU with exclusion radius for puffs and waves
- IoU of puff without holes
- plain precision and recall for sparks
- precision and recall for sparks with different threshold on puffs

Predictions and training annotations are saved in
`.\trainings_validation\{training_name}`

Movies and raw annotations are saved in `.\trainings_validation`

Metrics will be saved in `.\trainings_validation\{training_name}`

Remark: does not work (yet) for models using temporal reduction!
'''

import numpy as np
import glob
import os
import imageio
from collections import defaultdict
import pprint
import json

from scipy.ndimage.morphology import binary_dilation, binary_erosion
from sklearn.metrics import jaccard_score, f1_score

import pandas as pd
import matplotlib.pyplot as plt

import unet
from metrics_tools import (correspondences_precision_recall,
                           Metrics,
                           reduce_metrics,
                           empty_marginal_frames,
                           process_spark_prediction,
                           process_puff_prediction,
                           process_wave_prediction,
                           compute_puff_wave_metrics,
                           compute_average_puff_wave_metrics,
                           write_videos_on_disk,
                           get_sparks_locations_from_mask,
                           compute_prec_rec,
                           reduce_metrics_thresholds
                          )
from dataset_tools import load_annotations, load_predictions, load_movies_ids


# get current directory
BASEDIR = os.path.abspath('')

# set metrics to compute
compute_puff_wave_ious = False
compute_joined_puff_wave_ious = False
compute_joined_spark_puff_ious = True
compute_joined_all_classes_ious = True
compute_puff_no_holes_ious = False
compute_sparks_prec_rec = False
compute_sparks_on_puffs_prec_rec = False


############################### LOAD STORED DATA ###############################

# Select predictions to load
training_names = [#"256_long_chunks_ubelix",
                  "focal_loss_gamma_5_ubelix",
                  "focal_loss_new_sparks_ubelix"
                  ]

epoch = 100000

# Set folder where data is loaded and saved
metrics_folder = "trainings_validation"
os.makedirs(metrics_folder, exist_ok=True)

# Load raw annotations (sparks unprocessed)
ys_all_trainings = load_annotations(metrics_folder, mask_names="mask")

# Load original movies
movie_names = ys_all_trainings.keys()
movie_path = os.path.join("..","data","raw_data_and_processing","original_movies")
movies = load_movies_ids(movie_path, movie_names)

# General parameters
ignore_frames = 6

# Load predictions and training annotations
# remark: does not work for models using temporal reduction !!!
ys = {} # contains annotations for each training
sparks = {} # contains sparks for each training
puffs = {} # contains puffs for each training
waves = {} # contains waves for each training

for training_name in training_names:
    print(f"Processing training name {training_name}...")
    print()
    # Import .tif files as numpy array
    data_folder = os.path.join(metrics_folder, training_name)
    t_ys, t_sparks, t_puffs, t_waves = load_predictions(training_name,
                                                        epoch,
                                                        data_folder)

    ys[training_name] = t_ys
    sparks[training_name] = t_sparks
    puffs[training_name] = t_puffs
    waves[training_name] = t_waves


################################################################################
########################## PUFFS AND WAVES METRICS #############################
################################################################################

######################### METRICS FOR PUFFS AND WAVES ##########################

'''
Metrics considered: iou index, accuracy, precision, recall.
'''

if compute_puff_wave_ious:
    print("Computing metrics for puffs and waves")

    # puffs and waves params
    t_detection = np.round(np.linspace(0,1,21),2)
    min_radius = [0,1,2,3,4,5,6,7,8,9,10]

    exclusion_radius = [0,1,2,3,4,5,6,7,8,9,10]

    ious_puffs_all_models = {} # training_name x t x min_r x exclusion_r x video_id x metrics
    ious_waves_all_models = {} # training_name x t x min_r x exclusion_r x video_id x metrics

    ious_puffs_all_models_average = {} # training_name x t x min_r x exclusion_r x metrics
    ious_waves_all_models_average = {} # training_name x t x min_r x exclusion_r x metrics

    for training_name in training_names:
        print(training_name)
        # get predictions
        puffs_training = puffs[training_name]
        waves_training = waves[training_name]

        # init empty dictionaires
        ious_puffs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        ious_waves = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

        for video_id in ys_all_trainings.keys():
            print("\tVideo name:", video_id)
            ys_sample = ys_all_trainings[video_id]
            puffs_sample = puffs_training[video_id]
            waves_sample = waves_training[video_id]

            # get ignore mask (events labelled with 4)
            ignore_mask = empty_marginal_frames(np.where(ys_sample==4,1,0),
                                                ignore_frames)

            # get binary ys and remove ignored frames
            ys_puffs_sample = empty_marginal_frames(np.where(ys_sample==3,1,0),
                                                    ignore_frames)
            ys_waves_sample = empty_marginal_frames(np.where(ys_sample==2,1,0),
                                                    ignore_frames)

            for t in t_detection:
                #print("\t\tDetection threshold:", t)
                for min_r in min_radius:
                    #print("\t\t\tMinimal radius:", min_r)
                    # get binary predictions and remove ignored frames
                    puffs_binary = process_puff_prediction(pred=puffs_sample,
                                                           t_detection=t,
                                                           min_radius=min_r,
                                                           ignore_frames=ignore_frames)
                    waves_binary = process_wave_prediction(waves_sample,
                                                           t_detection=t,
                                                           min_radius=min_r,
                                                           ignore_frames=ignore_frames)

                    # compute IoU for list of exclusion radius values
                    for exclusion_r in exclusion_radius:
                        #print("\t\t\t\tExclusion radius:", exclusion_r)
                        # puffs
                        metrics = compute_puff_wave_metrics(ys=ys_puffs_sample,
                                                            preds=puffs_binary,
                                                            exclusion_radius=exclusion_r,
                                                            ignore_mask=ignore_mask)
                        ious_puffs[t][min_r][exclusion_r][video_id] = metrics

                        # waves
                        metrics = compute_puff_wave_metrics(ys=ys_waves_sample,
                                                            preds=waves_binary,
                                                            exclusion_radius=exclusion_r,
                                                            ignore_mask=ignore_mask)
                        ious_waves[t][min_r][exclusion_r][video_id] = metrics

        ious_puffs_all_models[training_name] = ious_puffs
        ious_waves_all_models[training_name] = ious_waves

        # save ious dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "puff_wave_metrics")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"metrics_puffs.json"),"w") as f:
            json.dump(ious_puffs,f)

        with open(os.path.join(data_folder,"metrics_waves.json"),"w") as f:
            json.dump(ious_waves,f)

        # compute average over movies
        print("\tComputing average")
        ious_puffs_average = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        ious_waves_average = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for t in t_detection:
            for min_r in min_radius:
                for exclusion_r in exclusion_radius:
                    ious_puffs_all_movies = ious_puffs_all_models[training_name][t][min_r][exclusion_r]
                    ious_waves_all_movies = ious_waves_all_models[training_name][t][min_r][exclusion_r]
                    ious_puffs_average[t][min_r][exclusion_r] = compute_average_puff_wave_metrics(ious_puffs_all_movies)
                    ious_waves_average[t][min_r][exclusion_r] = compute_average_puff_wave_metrics(ious_waves_all_movies)

        ious_puffs_all_models_average[training_name] = ious_puffs_average
        ious_waves_all_models_average[training_name] = ious_waves_average

        # save average ious dictionaires on disk as json
        with open(os.path.join(data_folder,"metrics_puffs_average.json"),"w") as f:
            json.dump(ious_puffs_average,f)

        with open(os.path.join(data_folder,"metrics_waves_average.json"),"w") as f:
            json.dump(ious_waves_average,f)


##################### METRICS FOR JOINED PUFFS AND WAVES #######################

if compute_joined_puff_wave_ious:
    print("Computing metrics for joined puffs and waves")
    # Params are best for 'focal_loss_gamma_5_ubelix'

    # puffs and waves params
    t_detection = np.round(np.linspace(0,1,21),2)
    min_radius = [0,1,2,3,4,5,6,7,8,9,10]

    exclusion_radius = [0,1,2,3,4,5,6,7,8,9,10]

    ious_sum_all_models = {} # training_name x ...
    ious_sum_average_all_models = {} # training_name x ...

    for training_name in training_names:
        print(training_name)
        # get predictions
        puffs_training = puffs[training_name]
        waves_training = waves[training_name]

        # init empty dictionaires
        ious_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

        for video_id in ys_all_trainings.keys():
            print("\tVideo name:", video_id)
            ys_sample = ys_all_trainings[video_id]
            puffs_sample = puffs_training[video_id]
            waves_sample = waves_training[video_id]

            # sum puff and wave preds
            preds_sum_sample = puffs_sample + waves_sample

            # get binary ys and remove ignored frames
            ys_puffs_sample = empty_marginal_frames(np.where(ys_sample==3,1,0),
                                                    ignore_frames)
            ys_waves_sample = empty_marginal_frames(np.where(ys_sample==2,1,0),
                                                    ignore_frames)
            ys_sum_sample = np.logical_or(ys_puffs_sample, ys_waves_sample)

            # get ignore mask (events labelled with 4)
            ignore_mask = empty_marginal_frames(np.where(ys_sample==4,1,0),
                                                ignore_frames)

            for t in t_detection:
                #print("t_detection:",t)
                for min_r in min_radius:
                    #print("min_radius:",min_r)
                    # get binary mask
                    preds_sum_binary = process_puff_prediction(preds_sum_sample,
                                                               t,
                                                               min_r,
                                                               ignore_frames)

                    # compute IoU for some exclusion radius values
                    for exclusion_r in exclusion_radius:
                        #print("exclusion_radius:",exclusion_r)
                        #print("exclusion radius:", radius)
                        ious_sum[t][min_r][exclusion_r][video_id] = compute_puff_wave_metrics(ys_sum_sample,
                                                                                              preds_sum_binary,
                                                                                              exclusion_r,
                                                                                              ignore_mask)

        ious_sum_all_models[training_name] = ious_sum

        # compute average over movies
        print("\tComputing average")
        ious_sum_average = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for t in t_detection:
                for min_r in min_radius:
                    for exclusion_r in exclusion_radius:
                        ious_sum_all_movies = ious_sum_all_models[training_name][t][min_r][exclusion_r]
                        ious_sum_average[t][min_r][exclusion_r] = compute_average_puff_wave_metrics(ious_sum_all_movies)

        ious_sum_average_all_models[training_name] = ious_sum_average

        # save ious dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "puff_wave_metrics")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"metrics_joined_puffs_waves.json"),"w") as f:
            json.dump(ious_sum_all_models[training_name],f)

        # save average ious dictionaires on disk as json
        with open(os.path.join(data_folder,"metrics_joined_puffs_waves_average.json"),"w") as f:
            json.dump(ious_sum_average_all_models[training_name],f)


##################### METRICS FOR JOINED SPARKS AND PUFFS ######################

if compute_joined_spark_puff_ious:
    print("Computing metrics for joined sparks and puffs")

    # sparks and puffs params
    t_detection = np.round(np.linspace(0,1,21),2)
    min_radius = [0,1,2,3,4,5,6,7,8,9,10]

    exclusion_radius = [0,1,2,3,4,5,6,7,8,9,10]

    ious_sum_all_models = {} # training_name x ...
    ious_sum_average_all_models = {} # training_name x ...

    for training_name in training_names:
        print(training_name)
        # get predictions
        puffs_training = puffs[training_name]
        sparks_training = sparks[training_name]

        # init empty dictionaires
        ious_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

        for video_id in ys_all_trainings.keys():
            print("\tVideo name:", video_id)
            ys_sample = ys_all_trainings[video_id]
            puffs_sample = puffs_training[video_id]
            sparks_sample = sparks_training[video_id]

            # sum puff and spark preds
            preds_sum_sample = puffs_sample + sparks_sample

            # get binary ys and remove ignored frames
            ys_puffs_sample = empty_marginal_frames(np.where(ys_sample==3,1,0),
                                                    ignore_frames)
            ys_sparks_sample = empty_marginal_frames(np.where(ys_sample==1,1,0),
                                                    ignore_frames)
            ys_sum_sample = np.logical_or(ys_puffs_sample, ys_sparks_sample)

            # get ignore mask (events labelled with 4)
            ignore_mask = empty_marginal_frames(np.where(ys_sample==4,1,0),
                                                ignore_frames)

            for t in t_detection:
                #print("t_detection:",t)
                for min_r in min_radius:
                    #print("min_radius:",min_r)
                    # get binary mask
                    preds_sum_binary = process_puff_prediction(preds_sum_sample,
                                                               t,
                                                               min_r,
                                                               ignore_frames)

                    # compute IoU for some exclusion radius values
                    for exclusion_r in exclusion_radius:
                        #print("exclusion_radius:",exclusion_r)
                        #print("exclusion radius:", radius)
                        ious_sum[t][min_r][exclusion_r][video_id] = compute_puff_wave_metrics(ys_sum_sample,
                                                                                              preds_sum_binary,
                                                                                              exclusion_r,
                                                                                              ignore_mask)

        ious_sum_all_models[training_name] = ious_sum

        # compute average over movies
        print("\tComputing average")
        ious_sum_average = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for t in t_detection:
                for min_r in min_radius:
                    for exclusion_r in exclusion_radius:
                        ious_sum_all_movies = ious_sum_all_models[training_name][t][min_r][exclusion_r]
                        ious_sum_average[t][min_r][exclusion_r] = compute_average_puff_wave_metrics(ious_sum_all_movies)

        ious_sum_average_all_models[training_name] = ious_sum_average

        # save ious dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "puff_wave_metrics")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"metrics_joined_puffs_sparks.json"),"w") as f:
            json.dump(ious_sum_all_models[training_name],f)

        # save average ious dictionaires on disk as json
        with open(os.path.join(data_folder,"metrics_joined_puffs_sparks_average.json"),"w") as f:
            json.dump(ious_sum_average_all_models[training_name],f)


######################## METRICS FOR ALL CLASSES JOINED ########################

if compute_joined_all_classes_ious:
    print("Computing metrics for all classes joined")

    # all classes params
    t_detection = np.round(np.linspace(0,1,21),2)
    min_radius = [0,1,2,3,4,5,6,7,8,9,10]

    exclusion_radius = [0,1,2,3,4,5,6,7,8,9,10]

    ious_sum_all_models = {} # training_name x ...
    ious_sum_average_all_models = {} # training_name x ...

    for training_name in training_names:
        print(training_name)
        # get predictions
        puffs_training = puffs[training_name]
        waves_training = waves[training_name]
        sparks_training = sparks[training_name]

        # init empty dictionaires
        ious_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

        for video_id in ys_all_trainings.keys():
            print("\tVideo name:", video_id)
            ys_sample = ys_all_trainings[video_id]
            puffs_sample = puffs_training[video_id]
            waves_sample = waves_training[video_id]
            sparks_sample = sparks_training[video_id]

            # sum all classes preds
            preds_sum_sample = puffs_sample + sparks_sample + waves_sample

            # get binary ys and remove ignored frames
            ys_puffs_sample = empty_marginal_frames(np.where(ys_sample==3,1,0),
                                                    ignore_frames)
            ys_sparks_sample = empty_marginal_frames(np.where(ys_sample==1,1,0),
                                                    ignore_frames)
            ys_waves_sample = empty_marginal_frames(np.where(ys_sample==2,1,0),
                                                    ignore_frames)
            ys_sum_sample = np.logical_or(ys_puffs_sample, ys_sparks_sample)
            ys_sum_sample = np.logical_or(ys_sum_sample, ys_waves_sample)

            # get ignore mask (events labelled with 4)
            ignore_mask = empty_marginal_frames(np.where(ys_sample==4,1,0),
                                                ignore_frames)

            for t in t_detection:
                #print("t_detection:",t)
                for min_r in min_radius:
                    #print("min_radius:",min_r)
                    # get binary mask
                    preds_sum_binary = process_puff_prediction(preds_sum_sample,
                                                               t,
                                                               min_r,
                                                               ignore_frames)

                    # compute IoU for some exclusion radius values
                    for exclusion_r in exclusion_radius:
                        #print("exclusion_radius:",exclusion_r)
                        #print("exclusion radius:", radius)
                        ious_sum[t][min_r][exclusion_r][video_id] = compute_puff_wave_metrics(ys_sum_sample,
                                                                                              preds_sum_binary,
                                                                                              exclusion_r,
                                                                                              ignore_mask)

        ious_sum_all_models[training_name] = ious_sum

        # compute average over movies
        print("\tComputing average")
        ious_sum_average = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for t in t_detection:
                for min_r in min_radius:
                    for exclusion_r in exclusion_radius:
                        ious_sum_all_movies = ious_sum_all_models[training_name][t][min_r][exclusion_r]
                        ious_sum_average[t][min_r][exclusion_r] = compute_average_puff_wave_metrics(ious_sum_all_movies)

        ious_sum_average_all_models[training_name] = ious_sum_average

        # save ious dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "puff_wave_metrics")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"metrics_joined_all_classes.json"),"w") as f:
            json.dump(ious_sum_all_models[training_name],f)

        # save average ious dictionaires on disk as json
        with open(os.path.join(data_folder,"metrics_joined_all_classes_average.json"),"w") as f:
            json.dump(ious_sum_average_all_models[training_name],f)


####################### METRICS FOR PUFFS WITHOUT HOLES ########################

if compute_puff_no_holes_ious:
    print("Computing metrics for puffs without holes")

    t_detection_puffs = np.round(np.linspace(0,1,21),2)
    min_radius_puffs = [0,1,2,3,4,5,6,7,8,9,10]

    ious_puffs_no_holes_all_models = {}
    ious_puffs_no_holes_average_all_models = {}

    for training_name in training_names:
        print(training_name)
        puffs_ious = defaultdict(lambda: defaultdict(dict))
        puffs_ious_all_movies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for movie_name in ys_all_trainings.keys():
            print("\tVideo name:", movie_name)
            # get puff binary annotations
            puffs_ys = ys[training_name][movie_name]
            puffs_ys = np.where(puffs_ys == 3, 1, 0)
            puffs_ys = empty_marginal_frames(puffs_ys, ignore_frames)

            # get ignore mask
            raw_ys = ys_all_trainings[movie_name]
            ignore_mask =  np.where(raw_ys==4,1,0)

            # get binary puff prediction
            puffs_pred = puffs[training_name][movie_name]

            for t in t_detection_puffs:
                for r in min_radius_puffs:
                    # remove holes (convex hull) & small events from puffs
                    puffs_pred_processed = process_puff_prediction(pred=puffs_pred,
                                                         t_detection=t,
                                                         min_radius=r,
                                                         ignore_frames=ignore_frames,
                                                         convex_hull=True
                                                        )

                    # compute IoU
                    puffs_ious_all_movies[t][r][movie_name] = compute_puff_wave_metrics(ys=puffs_ys,
                                                                                        preds=puffs_pred_processed,
                                                                                        exclusion_radius=0,
                                                                                        ignore_mask=ignore_mask)

        ious_puffs_no_holes_all_models[training_name] = puffs_ious_all_movies

        # save ious dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "puff_wave_metrics")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"metrics_puffs_no_holes.json"),"w") as f:
            json.dump(puffs_ious_all_movies,f)

        # compute average over movies
        print("\tComputing average")
        for t in t_detection_puffs:
            for r in min_radius_puffs:
                ious = puffs_ious_all_movies[t][r]
                puffs_ious[t][r] = compute_average_puff_wave_metrics(ious)

        ious_puffs_no_holes_average_all_models[training_name] = puffs_ious

        # save average ious dictionaires on disk as json
        with open(os.path.join(data_folder,"metrics_puffs_no_holes_average.json"),"w") as f:
            json.dump(puffs_ious,f)


################################################################################
################################ SPARKS METRICS ################################
################################################################################

# physiological params
PIXEL_SIZE = 0.2 # 1 pixel = 0.2 um x 0.2 um
MIN_DIST_XY = round(1.8 / PIXEL_SIZE) # min distance in space between sparks
TIME_FRAME = 6.8 # 1 frame = 6.8 ms
MIN_DIST_T = round(20 / TIME_FRAME) # min distance in time between sparks


###################### PLAIN SPARKS PRECISION AND RECALL #######################

if compute_sparks_prec_rec:
    print("Computing precision and recall for sparks")
    t_detection_sparks = np.round(np.linspace(0,1,21),2)
    min_radius_sparks = [0,1,2]

    # training name x min_r x video id x thresholds:
    prec_rec_sparks_all_trainings = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # training name x min_r x thresholds:
    prec_rec_sparks_avg = defaultdict(lambda: defaultdict(dict))
    #prec_avg = defaultdict(lambda: defaultdict(dict))
    #rec_avg = defaultdict(lambda: defaultdict(dict))

    for training_name in training_names:
        for movie_name in ys_all_trainings.keys():
            print("\tVideo name:", movie_name)
            # get preds
            sparks_sample = sparks[training_name][movie_name]
            # get annotations
            ys_sample = ys[training_name][movie_name]
            ys_raw = ys_all_trainings[movie_name]
            # get movie
            movie = movies[movie_name]

            # get binary ys
            ys_sparks_sample = np.where(ys_sample==1,1.0,0.0)

            # get ignore mask
            ignore_mask =  np.where(ys_raw==4,1,0)

            for min_r in min_radius_sparks:
                # compute precision and recall for some thresholds and remove ignored frames
                prec_rec_all_t = compute_prec_rec(annotations=ys_sparks_sample,
                                                  preds=sparks_sample,
                                                  movie=movie,
                                                  thresholds=t_detection_sparks,
                                                  ignore_frames=ignore_frames,
                                                  min_radius=min_r,
                                                  min_dist_xy=MIN_DIST_XY,
                                                  min_dist_t=MIN_DIST_T,
                                                  ignore_mask=ignore_mask
                                                 ) # dict indexed by threshold value
                prec_rec_sparks_all_trainings[training_name][min_r][movie_name] = prec_rec_all_t

        # compute average over all videos
        print("\tComputing average")
        for min_r in min_radius_sparks:
            prec_rec_all_videos = prec_rec_sparks_all_trainings[training_name][min_r]
            prec_rec_sparks_avg[training_name][min_r] = reduce_metrics_thresholds(prec_rec_all_videos)
            #prec_avg[training_name][min_r] = prec_rec_sparks_avg[training_name][min_r][1]
            #rec_avg[training_name][min_r] = prec_rec_sparks_avg[training_name][min_r][2]

        # save prec and rec dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "spark_prec_rec")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"prec_rec_sparks.json"),"w") as f:
            json.dump(prec_rec_sparks_all_trainings[training_name],f)

        with open(os.path.join(data_folder,"prec_rec_sparks_average.json"),"w") as f:
            json.dump(prec_rec_sparks_avg[training_name],f)


##################### SPARKS ON PUFFS PRECISION AND RECALL #####################

if compute_sparks_on_puffs_prec_rec:
    print("Computing precision and recall for sparks with adjusted threshold on puffs")

    t_puffs_lower = 0.3
    t_puffs_upper = [0.0,0.5,0.55,0.6,0.65] # = t detection puffs (0 for standard detection) (t included)

    t_detection_sparks = np.round(np.linspace(0,1,21),2) # for sum of puffs and sparks (t not included)

    min_radius_sparks = [0,1,2]

    prec_rec_on_puffs_all_trainings = {}
    prec_rec_on_puffs_all_movies_all_trainings = {}

    for training_name in training_names:
        print(training_name)
        prec_rec_on_puffs = defaultdict(lambda: defaultdict(dict))
        prec_rec_on_puffs_all_movies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for movie_name in ys_all_trainings.keys():
            print("\tVideo name:", movie_name)
            # open spark preds
            sparks_pred = sparks[training_name][movie_name]
            # open puff preds
            puffs_pred = puffs[training_name][movie_name]

            # get movie
            movie = movies[movie_name]

            # get binary annotation mask
            ys_sparks = ys[training_name][movie_name]
            binary_ys_sparks = np.where(ys_sparks==1,1.0,0.0)

            # get ignore mask
            ys_raw = ys_all_trainings[movie_name]
            ignore_mask =  np.where(ys_raw==4,1,0)

            for min_r in min_radius_sparks:
                for t_p in t_puffs_upper:
                    # process spark & puff preds together :
                    # compute region where 0.3 <= puffs <= 0.65
                    binary_puffs_sparks = np.logical_and(puffs_pred <= t_p,
                                                         puffs_pred >= t_puffs_lower)
                    # sum value of sparks and puffs in this region
                    sparks_pred_total = sparks_pred + binary_puffs_sparks * puffs_pred

                    # compute prec & rec for all t_detection_sparks
                    prec_rec_all_t = compute_prec_rec(annotations=binary_ys_sparks,
                                                      preds=sparks_pred_total,
                                                      movie=movie,
                                                      thresholds=t_detection_sparks,
                                                      ignore_frames=ignore_frames,
                                                      min_radius=min_r,
                                                      min_dist_xy=MIN_DIST_XY,
                                                      min_dist_t=MIN_DIST_T,
                                                      ignore_mask=ignore_mask
                                                      ) # dict of Metrics indexed by threshold value
                    #print(f"precision and recall for all t_detection_sparks: {prec_rec_all_t}")
                    prec_rec_on_puffs_all_movies[min_r][t_p][movie_name] = prec_rec_all_t

        prec_rec_on_puffs_all_movies_all_trainings[training_name] = prec_rec_on_puffs_all_movies

        # compute average over all videos
        print("\tComputing average")
        for min_r in min_radius_sparks:
            for t_p in t_puffs_upper:
                prec_rec_all_videos = prec_rec_on_puffs_all_movies[min_r][t_p]
                prec_rec_on_puffs[min_r][t_p] = reduce_metrics_thresholds(prec_rec_all_videos)


        prec_rec_on_puffs_all_trainings[training_name] = prec_rec_on_puffs

        # save prec and rec dictionaires on disk as json
        data_folder = os.path.join(metrics_folder, training_name, "spark_prec_rec")
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder,"prec_rec_sparks_on_puffs.json"),"w") as f:
            json.dump(prec_rec_on_puffs_all_movies,f)

        with open(os.path.join(data_folder,"prec_rec_sparks_on_puffs_average.json"),"w") as f:
            json.dump(prec_rec_on_puffs,f)
