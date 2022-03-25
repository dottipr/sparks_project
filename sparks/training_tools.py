'''
Functions that are used during the training of the neural network
'''

import os
import os.path
import time
import logging

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
#from focal_losses import focal_loss

from metrics_tools import *


import wandb



__all__ = ["training_step",
           "test_function",
           "test_function_fixed_t",
           "mycycle",
           "sampler"
           ]

logger = logging.getLogger(__name__)

# Make one step of the training (update parameters and compute loss)
def training_step(sampler, network, optimizer, device, criterion,
                  dataset_loader, ignore_frames, wandb_log):
    #start = time.time()

    network.train()

    x, y = sampler(dataset_loader)
    x = x.to(device)
    y = y.to(device)

    #print("X SHAPE", x.shape)
    #print("Y SHAPE", y.shape)

    # detect nan in tensors
    #if (torch.isnan(x).any() or torch.isnan(y).any()):
    #    logger.info(f"Detect nan in network input: {torch.isnan(x).any()}")
    #    logger.info(f"Detect nan in network annotation: {torch.isnan(y).any()}")

    y_pred = network(x[:, None])

    #Compute loss
    loss = criterion(y_pred[:,:,ignore_frames:-ignore_frames],
                     y[:,ignore_frames:-ignore_frames].long())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if wandb_log:
        wandb.log({"U-Net training loss": loss.item()})

    #end = time.time()
    #print(f"Runtime for 1 training step: {end-start}")

    return {"loss": loss.item()}



'''# Compute some metrics on the predictions of the network
def test_function(network, device, criterion, testing_datasets, logger,
                  summary_writer, thresholds, idx_fixed_threshold,
                  ignore_frames, wandb_log):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    # TODO: implementation not finished (using test_function_fixed_t temporarily)
    network.eval()

    duration = testing_datasets[0].duration
    step = testing_datasets[0].step
    half_overlap = (duration-step)//2 # to re-build videos from chunks

    # (duration-step) has to be even
    assert (duration-step)%2 == 0, "(duration-step) is not even"


    loss = 0.0
    metrics = [] # store metrics for each video

    for test_dataset in testing_datasets:
        xs = []
        ys = []
        preds = []

        with torch.no_grad():
            if (len(test_dataset)>1):
                x,y = test_dataset[0]
                xs.append(x[:-half_overlap])
                ys.append(y[:-half_overlap])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)

                pred = network(x[None, None])

                loss += criterion(pred[:,:,:-half_overlap],
                                 y[:,:-half_overlap].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,:-half_overlap])

                for i in range(1,len(test_dataset)-1):
                    x,y = test_dataset[i]
                    xs.append(x[half_overlap:-half_overlap])
                    ys.append(y[half_overlap:-half_overlap])

                    x = torch.Tensor(x).to(device)
                    y = torch.Tensor(y[None]).to(device)

                    pred = network(x[None, None])

                    loss += criterion(pred[:,:,half_overlap:-half_overlap],
                                     y[:,half_overlap:-half_overlap].long())

                    pred = pred[0].cpu().numpy()
                    preds.append(pred[:,half_overlap:-half_overlap])

                x,y = test_dataset[-1]
                xs.append(x[half_overlap:])
                ys.append(y[half_overlap:])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)

                pred = network(x[None, None])

                loss += criterion(pred[:,:,half_overlap:],
                                 y[:,half_overlap:].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,half_overlap:])
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

        if test_dataset.pad != 0:
            xs = xs[:-test_dataset.pad]
            ys = ys[:-test_dataset.pad]
            preds = preds[:,:-test_dataset.pad]

        # predictions have logarithmic values

        # save preds as videos
        #write_videos_on_disk(xs=xs,ys=ys,preds=preds,
                             training_name=training_name,
                             video_name=test_dataset.video_name,
                             path="predictions")


        # compute predicted sparks and correspondences
        sparks = np.exp(preds[1])
        #sparks = sparks[ignore_frames:-ignore_frames]
        #sparks = np.pad(sparks,((ignore_frames,),(0,),(0,)), mode='constant')
        sparks = empty_marginal_frames(sparks, ignore_frames)

        #min_radius = 3 # minimal "radius" of a valid event

        #coords_preds = process_spark_prediction(sparks,
        #                                        t_detection=(threshold),
        #                                        min_radius=min_radius)

        #sparks_mask_true = ys[ignore_frames:-ignore_frames]
        #sparks_mask_true = np.pad(sparks_mask_true,((ignore_frames,),(0,),(0,)),
        #                          mode='constant')
        sparks_mask_true = empty_marginal_frames(ys, ignore_frames)
        sparks_mask_true = np.where(sparks_mask_true==1, 1.0, 0.0)


        #coords_true = nonmaxima_suppression(sparks_mask_true)

        #metrics.append(Metrics(*correspondences_precision_recall(coords_true,
        #                                    coords_preds, match_distance=6)))


        metrics.append(compute_prec_rec(sparks_mask_true, sparks, thresholds))


    # Compute average validation loss
    loss = loss.item()
    loss /= len(testing_datasets)

    # Compute metrics comparing ys and y_preds
    _, precs, recs, a_u_c = reduce_metrics_thresholds(metrics)

    prec = precs[idx_fixed_threshold]
    rec = recs[idx_fixed_threshold]
    logger.info("\tPrecision: {:.4g}".format(prec))
    logger.info("\tRecall: {:.4g}".format(rec))
    #logger.info("\tArea under the curve: {:.4g}".format(a_u_c))
    logger.info("\tValidation loss: {:.4g}".format(loss))


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


    results = {"sparks/precision": prec,
               "sparks/recall": rec,
               #"sparks/area_under_curve": a_u_c,
               "validation_loss": loss}

    if wandb_log:
        wandb.log(results)

    return results'''

def test_function_fixed_t(network, device, criterion, testing_datasets, logger,
                          summary_writer, ignore_frames, wandb_log,
                          t_sparks, t_puffs, t_waves,
                          training_name, sparks_min_radius, puffs_min_radius,
                          waves_min_radius, temporal_reduction=False,
                          num_channels=1):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    # Compute precision and recall only for a fixed threshold
    # Compute IoU for puffs and waves classes
    # TODO: fix 'test_function' s.t. it works properly w.r.t. prec recall plot

    network.eval()

    duration = testing_datasets[0].duration
    step = testing_datasets[0].step
    half_overlap = (duration-step)//2 # to re-build videos from chunks

    assert (duration-step)%2 == 0, "(duration-step) is not even"

    if temporal_reduction:
        assert half_overlap % num_channels == 0, \
        "with temporal reduction half_overlap must be a multiple of num_channels"
        half_overlap_mask = half_overlap // num_channels
    else:
        half_overlap_mask = half_overlap

    #print(f"chunk duration = {duration}; step = {step}; half_overlap = {half_overlap}; half_overlap_mask = {half_overlap_mask}")

    loss = 0.0
    metrics = {} # store metrics for each video
    metrics['sparks'] = {}
    metrics['puffs'] = {}
    metrics['waves'] = {}

    for test_dataset in testing_datasets:

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
                #if (torch.isnan(x).any() or torch.isnan(y).any()):
                #    logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                #    logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")

                pred = network(x[None, None])

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
                    #if (torch.isnan(x).any() or torch.isnan(y).any()):
                    #    logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                    #    logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")

                    pred = network(x[None, None])
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
                #if (torch.isnan(x).any() or torch.isnan(y).any()):
                #    logger.info(f"Detect nan in network input (test): {torch.isnan(x).any()}")
                #    logger.info(f"Detect nan in network annotation (test): {torch.isnan(y).any()}")
                pred = network(x[None, None])

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
            if temporal_reduction:
                ys = ys[:-(test_dataset.pad // num_channels)]
                preds = preds[:,:-(test_dataset.pad // num_channels)]
            else:
                ys = ys[:-test_dataset.pad]
                preds = preds[:,:-test_dataset.pad]

        # predictions have logarithmic values
        #print("INPUT SHAPE", xs.shape)
        #print("MASK SHAPE", ys.shape)
        #print("OUTPUT SHAPE", preds.shape)

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
    _, precs, recs, f1_scores = reduce_metrics_thresholds(metrics['sparks'])

    prec = precs[t_sparks]
    rec = recs[t_sparks]
    f1_score = f1_scores[t_sparks]

    '''
    # TODO: not working properly
    # save precision recall plot (on disk and TB)
    figure = plt.figure()
    plt.plot(recs, precs, marker = '.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("Precision-recall plot")

    #print("RECALLS", recs)
    #print("PRECISIONS", precs)
    #print("ADDING FIGURE TO TENSORBOARD")
    summary_writer.add_figure("testing/sparks/prec_rec_plot", figure)
    figure.savefig("prec_rec_plot.png")
    '''

    # Puffs and waves metrics
    puffs_iou = sum(metrics['puffs'].values())/len(metrics['puffs'])
    waves_iou = sum(metrics['waves'].values())/len(metrics['waves'])

    logger.info("\tPrecision: {:.4g}".format(prec))
    logger.info("\tRecall: {:.4g}".format(rec))
    logger.info("\tF1 score: {:.4g}".format(f1_score))
    #logger.info("\tArea under the curve: {:.4g}".format(a_u_c))
    logger.info("\tPuffs IoU: {:.4g}".format(puffs_iou))
    logger.info("\tWaves IoU: {:.4g}".format(waves_iou))
    logger.info("\tValidation loss: {:.4g}".format(loss))

    results = {"sparks/precision": prec,
               "sparks/recall": rec,
               "sparks/f1_score": f1_score,
               #"sparks/area_under_curve": a_u_c,
               "puffs/iou": puffs_iou,
               "waves/iou": waves_iou,
               "validation_loss": loss}

    if wandb_log:
        wandb.log(results)

    return results

# Iterator (?) over the dataset
def mycycle(dataset_loader):
    while True:
        for x in dataset_loader:
            yield x

#_cycle = mycycle(dataset_loader)

def sampler(dataset_loader):
    return next(mycycle(dataset_loader))#(_cycle)
