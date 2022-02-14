#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import configparser
import glob

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import wandb

import unet
from dataset_tools import random_flip, compute_class_weights, weights_init
from datasets import SparkDataset, SparkTestDataset
from training_tools import training_step, test_function_fixed_t, sampler
from metrics_tools import take_closest
from focal_losses import FocalLoss
from architecture import TempRedUNet

BASEDIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")

    ############################# load config file #############################

    parser.add_argument(
        'config',
        type=str,
        help="Input config file, used to configure training"
    )
    args = parser.parse_args()

    CONFIG_FILE = os.path.join(BASEDIR, args.config)
    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")

    ############################## set parameters ##############################

    params = {}

    # general params
    params['name'] = c.get("general", "run_name", fallback="run") # Run name
    params['load_name'] = c.get("general", "load_run_name", fallback=None)

    # training params
    params['load_epoch'] = c.getint("state", "load_epoch", fallback=0)
    params['train_epochs'] = c.getint("training", "epochs", fallback=5000)
    params['training'] = c.getboolean("general", "training") # Run training procedure on data
    params['testing'] = c.getboolean("general", "testing") # Run training procedure on data
    params['loss_function'] = c.get("training", "criterion", fallback="nll_loss")

    # data params
    params['dataset_basedir'] = c.get("data", "relative_path")
    params['dataset_size'] = c.get("data", "size", fallback="full")
    params['batch_size'] = c.getint("general", "batch_size", fallback="1")
    params['data_duration'] = c.getint("data", "chunks_duration")
    params['data_step'] = c.getint("data", "step")
    params['ignore_frames_loss'] = c.getint("data", "ignore_frames_loss")
    params['only_sparks'] = c.getboolean("data", "only_sparks", fallback=False)

    # UNet params
    params['unet_steps'] = c.getint("network", "step")
    params['first_layer_channels'] = c.getint("network", "first_layer_channels")
    params['temporal_reduction'] = c.getboolean("network", "temporal_reduction", fallback=False)
    params['num_channels'] = c.getint("network", "num_channels", fallback=1)

    ############################# configure logger #############################

    level_map = {3: logging.DEBUG, 2: logging.INFO, 1: logging.WARNING, 0: logging.ERROR}
    log_level = level_map[c.getint("general", "verbosity", fallback="0")]
    log_handlers = (logging.StreamHandler(sys.stdout), )

    logfile = c.get("general", "logfile", fallback=None)

    if logfile:
        if not os.path.isdir(os.path.basename(logfile)):
            logger.info("Creating parent directory for logs")
            os.mkdir(os.path.basename(logfile))

        if os.path.isdir(logfile):
            logfile_path = os.path.abspath(os.path.join(logfile, f"{__name__}.log"))
        else:
            logfile_path = os.path.abspath(logfile)

        logger.info(f"Storing logs in {logfile_path}")
        file_handler = logging.RotatingFileHandler(
            filename=logfile_path,
            maxBytes=(1024 * 1024 * 8),  # 8 MB
            backupCount=4,
        )
        log_handlers += (file_handler, )

    logging.basicConfig(
        level=log_level,
        format='[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}',
        style='{',
        datefmt="%H:%M:%S",
        handlers=log_handlers)

    ############################# configure wandb ##############################

    if c.getboolean("general", "wandb_enable", fallback=False):
        wandb.init(project=c.get("general", "wandb_project_name"), name=params['name'])
        logging.getLogger('wandb').setLevel(logging.DEBUG)
        #wandb.save(CONFIG_FILE)

    ############################# print parameters #############################

    logger.info("Command parameters:")
    for k, v in params.items():
        logger.info(f"{k:>18s}: {v}")
        # TODO: AGGIUNGERE TUTTI I PARAMS NECESSARI DA PRINTARE

    ############################ configure datasets ############################

    # detect CUDA devices
    if c.getboolean("general", "cuda", fallback=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    # set if temporal reduction is used
    if params['temporal_reduction']:
        logger.info(f"Using temporal reduction with {params['num_channels']} channels")

    # normalize whole videos or chunks individually
    norm_video = c.getboolean("data", "norm_video", fallback=False)
    if norm_video:
        logger.info("Normalizing whole video instead of single chunks")

    # initialize training dataset
    dataset_map = {'full': "", 'small': 'small_dataset', 'minimal': 'very_small_dataset'}
    dataset_dir = dataset_map[params['dataset_size']]
    dataset_path = os.path.realpath(f"{BASEDIR}/{params['dataset_basedir']}/{dataset_dir}")
    assert os.path.isdir(dataset_path), f"\"{dataset_path}\" is not a directory"
    logger.info(f"Using {dataset_path} as dataset root path")
    dataset = SparkDataset(
        base_path=dataset_path,
        smoothing='2d',
        step=params['data_step'],
        duration=params['data_duration'],
        remove_background=c.getboolean("data", "remove_background"),
        temporal_reduction=params['temporal_reduction'],
        num_channels=params['num_channels'],
        normalize_video=norm_video,
        only_sparks=params['only_sparks']
    )

    # apply transforms
    dataset = unet.TransformedDataset(dataset, random_flip)

    logger.info(f"Samples in training dataset: {len(dataset)}")

    # initialize testing dataset
    test_file_names = sorted(glob.glob(f"{dataset_path}/videos_test/*[!_mask].tif"))
    testing_datasets = [
        SparkTestDataset(
            video_path=f,
            smoothing='2d',
            step=params['data_step'],
            duration=params['data_duration'],
            remove_background=c.getboolean("data", "remove_background"),
            temporal_reduction=params['temporal_reduction'],
            num_channels=params['num_channels'],
            normalize_video=norm_video,
            only_sparks=params['only_sparks']
        ) for f in test_file_names]

    for i, tds in enumerate(testing_datasets):
        logger.info(f"Testing dataset {i} contains {len(tds)} samples")

    # class weights
    else:
        class_weights = compute_class_weights(dataset)
    class_weights = torch.tensor(np.float32(class_weights))

    logger.info("Using class weights: {}".format(', '.join(str(w.item()) for w in class_weights)))

    # initialize data loaders
    dataset_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, num_workers=c.getint("training", "num_workers"))
    testing_dataset_loaders = [
        DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=c.getint("training", "num_workers"))
        for test_dataset in testing_datasets
    ]

    ############################## configure UNet ##############################

    unet_config = unet.UNetConfig(
        steps=params['unet_steps'],
        first_layer_channels=params['first_layer_channels'],
        num_classes=c.getint("network", "num_classes"),
        ndims=c.getint("network", "ndims"),
        dilation=c.getint("network", "dilation", fallback=1),
        border_mode=c.get("network", "border_mode"),
        batch_normalization=c.getboolean("network", "batch_normalization"),
        num_input_channels=params['num_channels'],
    )

    if not params['temporal_reduction']:
        network = unet.UNetClassifier(unet_config)
    else:
        assert params['data_duration'] % params['num_channels'] == 0, \
        "using temporal reduction chunks_duration must be a multiple of num_channels"
        network = TempRedUNet(unet_config)

    if device != "cpu":
        network = nn.DataParallel(network).to(device)

    if c.getboolean("general", "wandb_enable"):
        wandb.watch(network)

    if c.getboolean("network", "initialize_weights", fallback=False):
        network.apply(weights_init)

    ########################### set testing function ###########################

    #thresholds = np.linspace(0, 1, num=21) # thresholds for events detection
                                           # TODO: maybe change because
                                           # nonmaxima supression is computed
                                           # for every threshold (slow)
    fixed_threshold = c.getfloat("testing", "fixed_threshold", fallback = 0.9)
    #closest_t = take_closest(thresholds, fixed_threshold) # Compute idx of t in
                                                          # thresholds list that
                                                          # is closest to
                                                          # fixed_threshold
    #idx_fixed_t = list(thresholds).index(closest_t)

    ########################### initialize training ############################

    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    network.train()

    output_path = os.path.join(c.get("network", "output_relative_path"),
                               params['name'])
    logger.info(f"Output directory: {output_path}")

    summary_writer = SummaryWriter(os.path.join(output_path, "summary"),
                                   purge_step=0)

    if params['load_name'] != None:
        load_path = os.path.join(c.get("network", "output_relative_path"),
                                   params['load_name'])
        logger.info(f"Model loaded from directory: {load_path}")
    else:
        load_path = None


    if params['loss_function'] == "nll_loss":
        criterion = nn.NLLLoss(ignore_index=c.getint("data", "ignore_index"),
                               weight=class_weights.to(device))
    elif params['loss_function'] == "focal_loss":
        criterion = FocalLoss(reduction='mean',
                              ignore_index=c.getint("data", "ignore_index"),
                              alpha=class_weights)

    trainer = unet.TrainingManager(
        # training items
        training_step=lambda _: training_step(
            sampler,
            network,
            optimizer,
            device,
            criterion,
            dataset_loader,
            ignore_frames=params['ignore_frames_loss'],
            wandb_log=c.getboolean("general", "wandb_enable", fallback=False)
        ),
        save_every=c.getint("training", "save_every", fallback=5000),
        load_path=load_path,
        save_path=output_path,
        managed_objects=unet.managed_objects({
            'network': network,
            'optimizer': optimizer
        }),
        # testing items
        test_function=lambda _: test_function_fixed_t(
            network,
            device,
            criterion,
            testing_datasets,
            logger,
            summary_writer,
            fixed_threshold,
            #thresholds,
            #idx_fixed_t,
            ignore_frames=params['ignore_frames_loss'],
            wandb_log=c.getboolean("general", "wandb_enable", fallback=False),
            training_name=c.get("general", "run_name"),
            temporal_reduction=params['temporal_reduction'],
            num_channels=params['num_channels']
        ),
        test_every=c.getint("training", "test_every", fallback=1000),
        plot_every=c.getint("training", "plot_every", fallback=1000),
        summary_writer=summary_writer
    )

    ############################## start training ##############################

    if params['load_epoch'] != 0:
        trainer.load(params['load_epoch'])

    logger.info("Validate network before training")
    trainer.run_validation()

    if params['training']:
        logger.info("Starting training")
        trainer.train(params['train_epochs'], print_every=100)

    if params['testing']:
        logger.info("Starting final validation")
        trainer.run_validation()
