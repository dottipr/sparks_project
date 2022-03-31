'''
14.12.2021

Load a bunch of UNet trainings at given epochs and save their predictions in
folder `trainings_validation`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the results.

**Remark**: have to load dataset multiple times, because dataset configuration
might vary among different trainings
'''

import numpy as np
import glob
import sys
import os
import logging
import configparser

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter


import unet
from architecture import TempRedUNet
from metrics_tools import write_videos_on_disk
from datasets import SparkTestDataset

BASEDIR = os.path.dirname(os.path.realpath(__file__))

### Configure logger

logger = logging.getLogger(__name__)

log_level = logging.DEBUG
log_handlers = (logging.StreamHandler(sys.stdout), )

logging.basicConfig(
    level=log_level,
    format='[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}',
    style='{',
    datefmt="%H:%M:%S",
    handlers=log_handlers)

### Select trainings to load and config files to open

training_names = [#"temporal_reduction",
                  #"normalize_whole_video",
                  #"reduce_first_layer_channels_64",
                  #"256_long_chunks_physio",
                  #"256_long_chunks_64_step_physio",
                  #"temporal_reduction_ubelix",
                  #"256_long_chunks_ubelix",
                  #"focal_loss_ubelix",
                  #"focal_loss_gamma_5_ubelix",
                  #"focal_loss_new_sparks_ubelix",
                  #"pretrained_only_sparks_ubelix",
                  #"only_sparks_ubelix",
                  #"no_smoothing_physio",
                  "new_sparks_V3_physio"
                  ]
config_files = [#"config_temporal_reduction.ini",
                #"config_normalize_whole_video.ini",
                #"config_reduce_first_layer_channels.ini",
                #"config_256_long_chunks_physio.ini",
                #"config_256_long_chunks_64_step_physio.ini",
                #"config_temporal_reduction_ubelix.ini",
                #"config_256_long_chunks_ubelix.ini",
                #"config_focal_loss_ubelix.ini",
                #"config_pretrained_only_sparks_ubelix.ini",
                #"config_only_sparks_ubelix.ini",
                #"config_no_smoothing_physio.ini",
                "config_new_sparks_V3_physio.ini"
                 ]


### Select if prediction are computed for training or testing dataset

use_train_data = False
if use_train_data:
    logger.info("Predict outputs for training data")
else:
    logger.info("Predict outputs for testing data")


### Configure output folder

if not use_train_data:
    metrics_folder = "trainings_validation"
else :
    metrics_folder = os.path.join("trainings_validation", "train_samples")

os.makedirs(metrics_folder, exist_ok=True)


### Configure config files folder

config_folder = "config_files"


### Detect GPU, if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
logger.info(f"Using device <<{device}>> with {n_gpus} GPUs")


## For all selected trainings, compute UNet outputs for test dataset

def get_preds(network, device, datasets, ignore_frames, temporal_reduction, num_channels):
    '''process al movies in the UNet and get all predictions and movies as numpy arrays'''

    network.eval()

    duration = datasets[0].duration
    step = datasets[0].step
    half_overlap = (duration-step)//2 # to re-build videos from chunks
    # (duration-step) has to be even
    assert (duration-step)%2 == 0, "(duration-step) is not even"

    if temporal_reduction:
        assert half_overlap % num_channels == 0, \
        "with temporal reduction half_overlap must be a multiple of num_channels"
        half_overlap_mask = half_overlap // num_channels
    else:
        half_overlap_mask = half_overlap

    if hasattr(testing_datasets[0], 'video_name'):
        xs_all_videos = {}
        ys_all_videos = {}
        preds_all_videos = {}
    else:
        xs_all_videos = []
        ys_all_videos = []
        preds_all_videos = []

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

                pred = network(x[None, None])
                pred = pred[0].cpu().numpy()
                preds.append(pred[:,:-half_overlap_mask])

                for i in range(1,len(test_dataset)-1):
                    x,y = test_dataset[i]
                    xs.append(x[half_overlap:-half_overlap])
                    ys.append(y[half_overlap_mask:-half_overlap_mask])

                    x = torch.Tensor(x).to(device)
                    y = torch.Tensor(y[None]).to(device)

                    pred = network(x[None, None])
                    pred = pred[0].cpu().numpy()
                    preds.append(pred[:,half_overlap_mask:-half_overlap_mask])

                x,y = test_dataset[-1]
                xs.append(x[half_overlap:])
                ys.append(y[half_overlap_mask:])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)

                pred = network(x[None, None])
                pred = pred[0].cpu().numpy()
                preds.append(pred[:,half_overlap_mask:])

            else:
                x,y = test_dataset[0]
                xs.append(x)
                ys.append(y)

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)

                pred = network(x[None, None])
                pred = pred[0].cpu().numpy()
                preds.append(pred)

        # concatenated frames and predictions for a single video:
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=1)

        if test_dataset.pad != 0:
            xs = xs[:-test_dataset.pad]
            if temporal_reduction:
                ys = ys[:-(test_dataset.pad // num_channels)]
                preds = preds[:,:-(test_dataset.pad // num_channels)]
            else:
                ys = ys[:-test_dataset.pad]
                preds = preds[:,:-test_dataset.pad]

        # if dataset has video_name attribute, save results as dictionaries
        if hasattr(test_dataset, 'video_name'):
            xs_all_videos[test_dataset.video_name] = xs
            ys_all_videos[test_dataset.video_name] = ys
            preds_all_videos[test_dataset.video_name] = preds
        else:
            xs_all_videos.append(xs)
            ys_all_videos.append(ys)
            preds_all_videos.append(preds)

    return xs_all_videos, ys_all_videos, preds_all_videos


for training_name, config_name in zip(training_names, config_files):

    logger.info(f"Processing training <<{training_name}>>...")

    ########################### open config file ###########################

    CONFIG_FILE = os.path.join(BASEDIR, config_folder, config_name)
    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"\tLoading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.info(f"\tNo config file found at {CONFIG_FILE}, trying to use fallback values.")

    ########################### epoch ###########################

    load_epoch = c.getint("testing", "load_epoch")

    ########################### parameters ###########################

    batch_size = c.getint("testing", "batch_size", fallback="1")
    ignore_frames = c.getint("data", "ignore_frames_loss")

    temporal_reduction = c.getboolean("network", "temporal_reduction", fallback=False)
    num_channels = c.getint("network", "num_channels", fallback=1) if temporal_reduction else 1

    ########################### load dataset ###########################

    dataset_map = {'full': "", 'small': 'small_dataset', 'minimal': 'very_small_dataset'}
    dataset_size = c.get("testing", "dataset_size")
    dataset_dir = dataset_map[dataset_size]

    dataset_basedir = c.get("data", "relative_path")
    dataset_path = os.path.realpath(f"{BASEDIR}/{dataset_basedir}/{dataset_dir}")

    logger.info(f"\tUsing {dataset_size} dataset located in {dataset_path}")

    if not use_train_data:
        pattern_test_filenames = os.path.join(f"{dataset_path}","videos_test",
                                               "[0-9][0-9]_video.tif")
    else:
        pattern_test_filenames = os.path.join(f"{dataset_path}","videos",
                                               "[0-9][0-9]_video.tif")

    test_filenames = sorted(glob.glob(pattern_test_filenames))

    # create dataset
    testing_datasets = [
        SparkTestDataset(
            video_path=f,
            smoothing='2d',
            step=c.getint("data", "step"),
            duration=c.getint("data", "chunks_duration"),
            remove_background=c.get("data", "remove_background"),
            temporal_reduction=temporal_reduction,
            num_channels=num_channels
        ) for f in test_filenames]

    for i, tds in enumerate(testing_datasets):
        logger.info(f"\t\tTesting dataset {i} contains {len(tds)} samples")

    # dataloader
    testing_dataset_loaders = [
        DataLoader(test_dataset,
                   batch_size=c.getint("testing", "batch_size", fallback="1"),
                   shuffle=False,
                   num_workers=c.getint("training", "num_workers"))
        for test_dataset in testing_datasets
    ]

    ########################### configure UNet ###########################

    unet_config = unet.UNetConfig(
        steps=c.getint("network", "step"),
        first_layer_channels=c.getint("network", "first_layer_channels"),
        num_classes=c.getint("network", "num_classes"),
        ndims=c.getint("network", "ndims"),
        dilation=c.getint("network", "dilation", fallback=1),
        border_mode=c.get("network", "border_mode"),
        batch_normalization=c.getboolean("network", "batch_normalization"),
        num_input_channels=num_channels,
    )
    if not temporal_reduction:
        network = unet.UNetClassifier(unet_config)
    else:
        assert c.getint("data", "chunks_duration") % num_channels == 0, \
        "using temporal reduction chunks_duration must be a multiple of num_channels"
        network = TempRedUNet(unet_config)

    network = nn.DataParallel(network).to(device)

    ########################### load trained UNet params ###########################

    output_path = os.path.join(c.get("network", "output_relative_path"), training_name)
    logger.info(f"Saved model path: {output_path}")
    summary_writer = SummaryWriter(os.path.join(output_path, "summary"), purge_step=0)

    trainer = unet.TrainingManager(
            # training items
            training_step = None,
            save_path=output_path,
            managed_objects=unet.managed_objects({'network': network}),
            summary_writer=summary_writer
        )

    logger.info(f"\tLoading training <<{training_name}>> at epoch {load_epoch}...")
    trainer.load(load_epoch)
    logger.info(f"\tLoaded training located in <<{output_path}>>")

    ########################### run dataset in UNet ###########################

    logger.info(f"\tProcessing samples in UNet...")
    _, ys, preds = get_preds(network,
                             device,
                             testing_datasets,
                             ignore_frames,
                             temporal_reduction,
                             num_channels
                            )
    # ys and preds are dictionaries

    ########################### save preds on disk ###########################

    logger.info(f"\tSaving annotations and predictions on disk...")
    save_folder = os.path.join(metrics_folder, training_name)
    os.makedirs(save_folder, exist_ok=True)

    for (video_name, pred), (_, ys) in zip(preds.items(), ys.items()):
        new_video_name = str(load_epoch) + "_" + video_name

        # preds are in logarithmic scale -> exp computed while writing on disk
        write_videos_on_disk(training_name=training_name, video_name=new_video_name,
                             path=save_folder, preds=pred, ys=ys
                            )

    logger.info(f"Computed predicitions for training <<{training_name}>>")
