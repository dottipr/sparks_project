'''
Functions that are used during the training of the neural network
'''

import os
import os.path
import time
import logging
import pprint
import wandb

import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
from scipy import ndimage as ndi

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

#torch.__file__
import unet
#from other_losses import focal_loss

from metrics_tools import *

__all__ = ["training_step",
           "test_function",
           "test_function_fixed_t",
           "mycycle",
           "sampler"
           ]

logger = logging.getLogger(__name__)


################################ training step #################################

# Make one step of the training (update parameters and compute loss)
def training_step(sampler, network, optimizer, device, criterion,
                  dataset_loader, ignore_frames, wandb_log):
    #start = time.time()

    network.train()

    x, y = sampler(dataset_loader)
    x = x.to(device) # [1, 256, 64, 512]
    y = y.to(device) # [1, 256, 64, 512]

    # detect nan in tensors
    #if (torch.isnan(x).any() or torch.isnan(y).any()):
    #    logger.info(f"Detect nan in network input: {torch.isnan(x).any()}")
    #    logger.info(f"Detect nan in network annotation: {torch.isnan(y).any()}")

    y_pred = network(x[:, None]) # [1, 4, 256, 64, 512]

    #Compute loss
    loss = criterion(y_pred[...,ignore_frames:-ignore_frames],
                     y[...,ignore_frames:-ignore_frames].long())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if wandb_log:
        wandb.log({"U-Net training loss": loss.item()})

    #end = time.time()
    #print(f"Runtime for 1 training step: {end-start}")

    return {"loss": loss.item()}

################################ test function #################################

# function to run a test sample (i.e., a test dataset) in the UNet
def get_preds(network, test_dataset, compute_loss, device,
              criterion = None, detect_nan = False):

    # check if function parameters are correct
    if compute_loss:
        assert criterion is not None, "provide criterion if computing loss"
        loss = 0.

    assert (test_dataset.duration-test_dataset.step)%2 == 0, "(duration-step) is not even"
    half_overlap = (test_dataset.duration-test_dataset.step)//2
    # to re-build videos from chunks

    # adapt half_overlap duration if using temporal reduction
    if test_dataset.temporal_reduction:
        assert half_overlap % test_dataset.num_channels == 0, \
        "with temporal reduction half_overlap must be a multiple of num_channels"
        half_overlap_mask = half_overlap // test_dataset.num_channels
    else:
        half_overlap_mask = half_overlap

    #print(f"chunk duration = {duration}; step = {step}; half_overlap = {half_overlap}; half_overlap_mask = {half_overlap_mask}")

    xs = []
    ys = []
    preds = []

    with torch.no_grad():
        if (len(test_dataset)>1):
            x,y = test_dataset[0]
            xs.append(x[:-half_overlap])
            ys.append(y[:-half_overlap_mask])

            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y[None]).to(device)
            #print("X SHAPE", x.shape)
            #print("Y SHAPE", y.shape)

            # detect nan in tensors
            if detect_nan:
                if (torch.isnan(x).any() or torch.isnan(y).any()):
                    logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                    logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")

            pred = network(x[None, None])

            if compute_loss:
                loss += criterion(pred[:,:,:-half_overlap_mask],
                                  y[:,:-half_overlap_mask].long())

            pred = pred[0].cpu().numpy()
            preds.append(pred[:,:-half_overlap_mask])

            for i in range(1,len(test_dataset)-1):
                x,y = test_dataset[i]
                xs.append(x[half_overlap:-half_overlap])
                ys.append(y[half_overlap_mask:-half_overlap_mask])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                #print("X SHAPE", x.shape)
                #print("Y SHAPE", y.shape)

                # detect nan in tensors
                if detect_nan:
                    if (torch.isnan(x).any() or torch.isnan(y).any()):
                        logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                        logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")

                pred = network(x[None, None])

                if compute_loss:
                    loss += criterion(pred[:,:,half_overlap_mask:-half_overlap_mask],
                                      y[:,half_overlap_mask:-half_overlap_mask].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,half_overlap_mask:-half_overlap_mask])

            x,y = test_dataset[-1]
            xs.append(x[half_overlap:])
            ys.append(y[half_overlap_mask:])

            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y[None]).to(device)
            #print("X SHAPE", x.shape)
            #print("Y SHAPE", y.shape)

            # detect nan in tensors
            if detect_nan:
                if (torch.isnan(x).any() or torch.isnan(y).any()):
                    logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                    logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")

            pred = network(x[None, None])

            if compute_loss:
                loss += criterion(pred[:,:,half_overlap_mask:],
                                  y[:,half_overlap_mask:].long())

            pred = pred[0].cpu().numpy()
            preds.append(pred[:,half_overlap_mask:])
        else:
            x,y = test_dataset[0]
            xs.append(x)
            ys.append(y)

            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y[None]).to(device)
            pred = network(x[None, None])

            loss += criterion(pred, y.long())

            pred = pred[0].cpu().numpy()
            preds.append(pred)

    # concatenated frames and predictions for a single video:
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=1)

    #print("MASK OUTPUT SHAPE BEFORE REMOVING PADDING", ys.shape)
    #print("MASK PADDING", test_dataset.pad)
    #print("REMOVED FRAMES", test_dataset.pad // num_channels)

    if test_dataset.pad != 0:
        xs = xs[:-test_dataset.pad]
        if test_dataset.temporal_reduction:
            ys = ys[:-(test_dataset.pad // test_dataset.num_channels)]
            preds = preds[:,:-(test_dataset.pad // test_dataset.num_channels)]
        else:
            ys = ys[:-test_dataset.pad]
            preds = preds[:,:-test_dataset.pad]

    # predictions have logarithmic values
    #print("INPUT SHAPE", xs.shape)
    #print("MASK SHAPE", ys.shape)
    #print("OUTPUT SHAPE", preds.shape)

    if compute_loss:
        # divide loss by number of samples in test_dataset
        loss = loss.item()
        loss /= len(test_dataset)
        return xs, ys, preds, loss
    else:
        return xs, ys, preds


'''def test_function_fixed_t(network, device, criterion, testing_datasets, logger,
                          summary_writer, ignore_frames, wandb_log,
                          t_sparks, t_puffs, t_waves, training_name,
                          sparks_min_radius, puffs_min_radius, waves_min_radius):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    # Compute precision and recall only for a fixed threshold
    # Compute IoU for puffs and waves classes
    # TODO: fix 'test_function' s.t. it works properly w.r.t. prec recall plot
    network.eval()




    loss = 0.0
    metrics = {} # store metrics for each video
    metrics['sparks'] = {}
    metrics['puffs'] = {}
    metrics['waves'] = {}

    # initialize class attributes that are shared by all datasets
    sparks_type = testing_datasets[0].sparks_type
    temporal_reduction = testing_datasets[0].temporal_reduction
    if temporal_reduction:
        num_channels = testing_datasets[0].num_channels

    for test_dataset in testing_datasets:

        # run sample in UNet
        xs, ys, preds, loss = get_preds(network=network,
                                        test_dataset=test_dataset,
                                        compute_loss=True,
                                        device=device,
                                        half_overlap=half_overlap,
                                        criterion=criterion,
                                        detect_nan=False)

        # save preds as videos
        write_videos_on_disk(xs=xs,ys=ys,preds=preds,
                             training_name=training_name,
                             video_name=test_dataset.video_name,
                             path="predictions"
                             )

        ################### compute metrics for single video ###################

        # clean annotations
        ys = empty_marginal_frames(ys, ignore_frames)
        # get ignore mask ( = events labelled with 4)
        ignore_mask = np.where(ys==4,1,0)

        # Sparks metrics

        sparks = np.exp(preds[1])  # preds
        sparks_true = np.where(ys==1, 1.0, 0.0) # annotations

        if sparks_type == 'peaks':
            sparks_prec_rec = compute_prec_rec(annotations=sparks_true,
                                               preds=sparks,
                                               movie=xs,
                                               thresholds=[t_sparks],
                                               ignore_frames=ignore_frames,
                                               min_radius=sparks_min_radius)
            # Remark: dipending on normalisation, xs could have amplitude jumps
            # when chunks changes !! (so metrics are not 100% accurate)
            # This is not a problem when using the original movie
            metrics['sparks'][test_dataset.video_name] = sparks_prec_rec
        elif sparks_type == 'raw':
            sparks_binary = process_puff_prediction(pred=sparks,
                                                    t_detection=t_sparks,
                                                    min_radius=sparks_min_radius,
                                                    ignore_frames=ignore_frames)
            sparks_true = np.where(ys==1, 1, 0) # annotations
            sparks_iou = compute_puff_wave_metrics(ys=sparks_true,
                                                   preds=sparks_binary,
                                                   exclusion_radius=0,
                                                   ignore_mask=ignore_mask)['iou']
            metrics['sparks'][test_dataset.video_name] = sparks_iou

        # Puffs & waves metrics

        waves = np.exp(preds[2]) # preds
        waves_binary = process_wave_prediction(pred=waves,
                                               t_detection=t_waves,
                                               min_radius=waves_min_radius,
                                               ignore_frames=ignore_frames)
        waves_true = np.where(ys==2, 1, 0) # annotations
        waves_iou = compute_puff_wave_metrics(ys=waves_true,
                                              preds=waves_binary,
                                              exclusion_radius=0,
                                              ignore_mask=ignore_mask)['iou']
        metrics['waves'][test_dataset.video_name] = waves_iou

        puffs = np.exp(preds[3]) # preds
        puffs_binary = process_puff_prediction(pred=puffs,
                                               t_detection=t_puffs,
                                               min_radius=puffs_min_radius,
                                               ignore_frames=ignore_frames)
        puffs_true = np.where(ys==3, 1, 0) # annotations
        puffs_iou = compute_puff_wave_metrics(ys=puffs_true,
                                              preds=puffs_binary,
                                              exclusion_radius=0,
                                              ignore_mask=ignore_mask)['iou']
        metrics['puffs'][test_dataset.video_name] = puffs_iou

    # Compute average validation loss
    loss = loss.item()
    loss /= len(testing_datasets)

    ############################## reduce metrics ##############################

    # Sparks metrics
    if sparks_type == 'peaks':
        _, precs, recs, f1_scores = reduce_metrics_thresholds(metrics['sparks'])

        prec = precs[t_sparks]
        rec = recs[t_sparks]
        f1_score = f1_scores[t_sparks]

        # TODO: not working properly
        # save precision recall plot (on disk and TB)
        #figure = plt.figure()
        #plt.plot(recs, precs, marker = '.')
        #plt.xlim([0,1])
        #plt.ylim([0,1])
        #plt.xlabel('recall')
        #plt.ylabel('precision')
        #plt.title("Precision-recall plot")

        #print("RECALLS", recs)
        #print("PRECISIONS", precs)
        #print("ADDING FIGURE TO TENSORBOARD")
        #summary_writer.add_figure("testing/sparks/prec_rec_plot", figure)
        #figure.savefig("prec_rec_plot.png")
    elif sparks_type == 'raw':
        sparks_iou = sum(metrics['sparks'].values())/len(metrics['sparks'])

    # Puffs and waves metrics
    puffs_iou = sum(metrics['puffs'].values())/len(metrics['puffs'])
    waves_iou = sum(metrics['waves'].values())/len(metrics['waves'])

    if sparks_type == 'peaks':
        logger.info("\tPrecision: {:.4g}".format(prec))
        logger.info("\tRecall: {:.4g}".format(rec))
        logger.info("\tF1 score: {:.4g}".format(f1_score))
        #logger.info("\tArea under the curve: {:.4g}".format(a_u_c))
    if sparks_type == 'raw':
        logger.info("\tSparks IoU: {:.4g}".format(sparks_iou))

    logger.info("\tPuffs IoU: {:.4g}".format(puffs_iou))
    logger.info("\tWaves IoU: {:.4g}".format(waves_iou))
    logger.info("\tValidation loss: {:.4g}".format(loss))

    if sparks_type == 'peaks':
        results = {"sparks/precision": prec,
                   "sparks/recall": rec,
                   "sparks/f1_score": f1_score,
                   #"sparks/area_under_curve": a_u_c,
                   "puffs/iou": puffs_iou,
                   "waves/iou": waves_iou,
                   "validation_loss": loss}
    elif sparks_type == 'raw':
        results = {"sparks/iou": sparks_iou,
                   "puffs/iou": puffs_iou,
                   "waves/iou": waves_iou,
                   "validation_loss": loss}

    if wandb_log:
        wandb.log(results)

    return results'''

def test_function(network, device, criterion, ignore_frames, testing_datasets,
                  logger, wandb_log, training_name):
    '''
    Validate UNet during training.
    Output segmentation is computed using argmax values (to avoid using
    thresholds).
    Not using minimal radius to remove small events (yet).

    network:            the model being trained
    device:             current device
    criterion:          loss function to be computed on the validation set
    testing_datasets:   list of SparkTestDataset instances
    logger:             logger to output results in the terminal
    ignore_frames:      frames ignored by the loss function
    wandb_log:          logger to store results on wandb
    training_name:      training name used to save predictions on disk

    '''
    network.eval()

    pixel_based_results = {} # store metrics for each video
    pixel_based_results['sparks'] = {} # movie name x {tp, tn, fp, fn}
    pixel_based_results['puffs'] = {} # movie name x {tp, tn, fp, fn}
    pixel_based_results['waves'] = {} # movie name x {tp, tn, fp, fn}

    spark_peaks_results = {} # store metrics for each video
    spark_peaks_results['sparks'] = {}  # movie name x {tp, tp_fp, tp_fn}

    # initialize class attributes that are shared by all datasets
    sparks_type = testing_datasets[0].sparks_type
    temporal_reduction = testing_datasets[0].temporal_reduction
    if temporal_reduction:
        num_channels = testing_datasets[0].num_channels

    min_dist_xy = testing_datasets[0].min_dist_xy
    min_dist_t = testing_datasets[0].min_dist_t

    for test_dataset in testing_datasets:
        # run sample in UNet
        xs, ys, preds, loss = get_preds(network=network,
                                        test_dataset=test_dataset,
                                        compute_loss=True,
                                        device=device,
                                        criterion=criterion,
                                        detect_nan=False)
        # preds is a list [sparks, waves, puffs]

        # save preds as videos
        write_videos_on_disk(xs=xs,ys=ys,preds=preds,
                             training_name=training_name,
                             video_name=test_dataset.video_name,
                             path="predictions"
                             )

        ################## compute metrics for current video ###################

        # get annotations as a dictionary
        ys_classes = {'sparks': np.where(ys==1,1,0),
                      'puffs': np.where(ys==3,1,0),
                      'waves': np.where(ys==2,1,0)}

        # get pixels labelled with 4
        # TODO: togliere ignored frames ??????????????????????????????
        ignore_mask = np.where(ys==4,1,0)

        # compute exp of predictions
        preds = np.exp(preds)

        # get segmented output using argmax (as a dict)
        argmax_preds = get_argmax_segmented_output(preds=preds,
                                                   get_classes=True)[0]

        # get list of classes
        classes_list = argmax_preds.keys()

        for event_class in classes_list:

            ################## COMPUTE PIXEL-BASED RESULTS #####################

            class_preds = argmax_preds[event_class]
            class_ys = ys_classes[event_class]

            # can remove marginal frames, since considering pixel-based metrics
            class_preds = empty_marginal_frames(class_preds, ignore_frames)
            class_ys = empty_marginal_frames(class_ys, ignore_frames)

            tp,tn,fp,fn = compute_puff_wave_metrics(ys=class_ys,
                                                    preds=class_preds,
                                                    exclusion_radius=0,
                                                    ignore_mask=ignore_mask,
                                                    sparks=(event_class=='sparks'),
                                                    results_only=True)

            pixel_based_results[event_class][test_dataset.video_name] = {'tp':tp,
                                                                         'tn':tn,
                                                                         'fp':fp,
                                                                         'fn':fn}

            ################### COMPUTE SPARK PEAKS RESULTS ####################

            if event_class == 'sparks':
                # extract peak locations from annotations used during training
                if sparks_type == 'peaks':
                    coords_true = get_sparks_locations_from_mask(mask=ys,
                                                                 min_dist_xy=min_dist_xy,
                                                                 min_dist_t=min_dist_t,
                                                                 ignore_frames=ignore_frames)
                elif sparks_type == 'raw':
                    coords_true = get_new_mask(video=xs,
                                               mask=ys,
                                               min_dist_xy=min_dist_xy,
                                               min_dist_t=min_dist_t,
                                               return_loc=True,
                                               ignore_frames=ignore_frames)
                else:
                    logger.warn("WARNING: something is wrong...")

                # get sparks preds
                class_preds = argmax_preds[event_class]

                # get predicted peaks locations
                coords_pred = nonmaxima_suppression(img=xs,
                                                    maxima_mask=class_preds,
                                                    min_dist_xy=min_dist_xy,
                                                    min_dist_t=min_dist_t,
                                                    return_mask=False,
                                                    threshold=0,
                                                    sigma=2) # TODO: maybe need to change sigma!!!!!

                # remove events ignored by loss function in preds
                if ignore_frames > 0:
                    mask_duration = ys.shape[0]
                    coords_pred = empty_marginal_frames_from_coords(coords=coords_pred,
                                                                    n_frames=ignore_frames,
                                                                    duration=mask_duration)

                # get results as a dict {tp, tp_fp, tp_fn}
                nb_results = correspondences_precision_recall(coords_real=coords_true,
                                                              coords_pred=coords_pred,
                                                              match_distance_xy=min_dist_xy,
                                                              match_distance_t=min_dist_t,
                                                              return_nb_results=True)

                spark_peaks_results[event_class][test_dataset.video_name] = nb_results



    ############################## reduce metrics ##############################

    metrics = {}

    # Compute average validation loss
    loss /= len(testing_datasets)

    metrics['validation_loss'] = loss
    logger.info(f"\tvalidation loss: {loss:.4g}")

    for event_class in classes_list:

        #################### COMPUTE PIXEL-BASED RESULTS #######################
        '''
        Metrics that can be computed (raw sparks, puffs, waves):
        - Jaccard index (IoU)
        - Dice score
        - Precision & recall
        - F-score (e.g. beta = 0.5,1,2)
        - Accuracy (biased since background is predominant)
        - Matthews correlation coefficient (MCC)
        '''

        # compute average of all results
        res = {} # tp, tn, fp, fn
        for movie_name, movie_res in pixel_based_results[event_class].items():
            for r, val in movie_res.items():
                if r in res:
                    res[r] += val/len(testing_datasets)
                else:
                    res[r] = val/len(testing_datasets)

        iou = res['tp']/(res['tp']+res['fn']+res['fp']) if (res['tp']+res['fn']+res['fp']) != 0 else 1.0
        dice = res['tp']/(2*res['tp']+res['fn']+res['fp']) if (res['tp']+res['fn']+res['fp']) != 0 else 1.0
        prec = res['tp']/(res['tp']+res['fp']) if (res['tp']+res['fp']) != 0 else 1.0
        rec = res['tp']/(res['tp']+res['fn']) if (res['tp']+res['fn']) != 0 else 1.0
        accuracy = (res['tp']+res['tn'])/(res['tp']+res['tn']+res['fp']+res['fn'])
        mcc = (res['tp']*res['tn']-res['fp']*res['fn'])/np.sqrt((res['tp']+res['fp'])*(res['tp']+res['fn'])*(res['tn']+res['fp'])*(res['tn']+res['fn'])) if (res['tp']+res['fp'])*(res['tp']+res['fn'])*(res['tn']+res['fp'])*(res['tn']+res['fn']) != 0 else 0.0


        metrics[event_class+"/iou"] = iou
        metrics[event_class+"/dice"] = dice
        metrics[event_class+"/pixel_prec"] = prec
        metrics[event_class+"/pixel_rec"] = rec
        metrics[event_class+"/accuracy"] = accuracy
        metrics[event_class+"/mcc"] = mcc


        ################### COMPUTE SPARK PEAKS RESULTS ####################

        '''
        Metrics that can be computed (spark peaks):
        - Precision & recall
        - F-score (e.g. beta = 0.5,1,2)
        (- Matthews correlation coefficient (MCC))???
        '''

        if event_class == 'sparks':

            # compute average of all results
            res = {} # tp, tp_fn, tp_fp
            for movie_name, movie_res in spark_peaks_results[event_class].items():
                for r, val in movie_res.items():
                    if r in res:
                        res[r] += val/len(testing_datasets)
                    else:
                        res[r] = val/len(testing_datasets)

            prec = res['tp']/res['tp_fp'] if res['tp_fp'] != 0 else 1.0
            rec = res['tp']/res['tp_fn'] if res['tp_fn'] != 0 else 1.0

            metrics[event_class+"/precision"] = prec
            metrics[event_class+"/recall"] = rec

            betas = [0.5,1,2]
            for beta in betas:
                f_score = compute_f_score(prec, rec, beta)
                metrics[event_class+f"/f{beta}_score"] = f_score

    for metric, val in metrics.items():
        logger.info(f"\t{metric}: {val:.4g}")

    if wandb_log:
        wandb.log(metrics)

    return metrics

# Iterator (?) over the dataset
def mycycle(dataset_loader):
    while True:
        for x in dataset_loader:
            yield x

#_cycle = mycycle(dataset_loader)

def sampler(dataset_loader):
    return next(mycycle(dataset_loader))#(_cycle)
