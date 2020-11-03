#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import configparser

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import wandb

import unet
from dataset_tools import random_flip, compute_class_weights_puffs, weights_init
from datasets import MaskTEMPDataset, MaskTEMPTestDataset
from training_tools import training_step, test_function, sampler


BASEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASEDIR, "config.ini")

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")
    
    parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")

    parser.add_argument(
        '-v-', '--verbosity', 
        type=int, 
        default=c.getint("general", "verbosity", fallback="0"),
        help="Set verbosity level ([0, 3])"
    )

    parser.add_argument(
        '--logfile',
        type=str,
        default=None,
        help="In addition to stdout, store all logs in this file"
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        default=c.get("general", "run_name", fallback="run"),
        help="Run name"
    )

    parser.add_argument(
        '-T', '--training',
        action='store_true',
        default=c.getboolean("general", "training"),
        help="Run training procedure on data"
    )

    parser.add_argument(
        '-t', '--testing',
        action='store_true',
        default=c.getboolean("general", "testing"),
        help="Run training procedure on data"
    )

    parser.add_argument(
        '-l', '--load-epoch',
        type=int,
        default=c.getint("state", "load_epoch", fallback=0),
        help="Load the network state from this epoch"
    )

    parser.add_argument(
        '-e', '--train-epochs',
        type=int,
        default=c.getint("training", "epochs", fallback=5000),
        help="Train this many epochs"
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=c.getint("general", "batch_size", fallback="1"),
        help="Use this batch size for training & evaluation."
    )

    parser.add_argument(
        '-d', '--dataset-size', 
        type=str,
        choices=('full', 'small', 'minimal'),
        default=c.get("data", "size", fallback="full"),
        help="Use this extent of the dataset for testing during training"
    )

    parser.add_argument(
        '-w0', '--weight-background', 
        type=float,
        default=c.getfloat("data", "weight_background", fallback="1"),
        help="Weight proportion for background"
    )

    parser.add_argument(
        '-w1', '--weight-sparks', 
        type=float,
        default=c.getfloat("data", "weight_sparks", fallback="1"),
        help="Weight proportion for sparks"
    )

    parser.add_argument(
        '-w2', '--weight-waves', 
        type=float,
        default=c.getfloat("data", "weight_waves", fallback="1"),
        help="Weight proportion for waves"
    )

    parser.add_argument(
        '-w3', '--weight-puffs', 
        type=float,
        default=c.getfloat("data", "weight_puffs", fallback="1"),
        help="Weight proportion for puffs"
    )
    
    args = parser.parse_args()

    level_map = {3: logging.DEBUG, 2: logging.INFO, 1: logging.WARNING, 0: logging.ERROR}
    log_level = level_map[args.verbosity]
    log_handlers = (logging.StreamHandler(sys.stdout), )

    if args.logfile:
        if not os.path.isdir(os.path.basename(args.logfile)):
            logger.info("Creating parent directory for logs")
            os.mkdir(os.path.basename(args.logfile))

        if os.path.isdir(args.logfile):
            logfile_path = os.path.abspath(os.path.join(args.logfile, f"{__name__}.log"))
        else:
            logfile_path = os.path.abspath(args.logfile)

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

    logger.info("Command parameters:")
    for k, v in vars(args).items():
        logger.info(f"{k:>18s}: {v}")

    if c.getboolean("general", "wandb_enable", fallback="no"):
        wandb.init(project=c.get("general", "wandb_project_name"), name=args.name)

    # detect CUDA devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    # disable U-Net's logging to file
    #unet.config_logger("/dev/null")

    # initialize training dataset
    dataset_map = {'full': "", 'small': 'small_dataset', 'minimal': 'very_small_dataset'}
    dataset_dir = dataset_map[args.dataset_size]
    dataset_basedir = c.get("data", "relative_path")
    dataset_path = os.path.realpath(f"{BASEDIR}/{dataset_basedir}/{dataset_dir}")
    assert os.path.isdir(dataset_path), f"\"{dataset_path}\" is not a directory"
    logger.info(f"Using {dataset_path} as dataset root path")
    
    dataset = MaskTEMPDataset(
        base_path=dataset_path,
        smoothing='2d',
        step=c.getint("data", "step"),
        duration=c.getint("data", "chunks_duration"),
        remove_background=c.getboolean("data", "remove_background")
    )

    # apply transforms
    dataset = unet.TransformedDataset(dataset, random_flip)

    logger.info(f"Samples in training dataset: {len(dataset)}")

    # initialize testing dataset
    test_file_names = os.listdir(f"{dataset_path}/video_temp_test")
    # remove file extensions
    test_base_file_names = [base for base, ext in map(os.path.splitext, test_file_names)]
    testing_datasets = [
        MaskTEMPTestDataset(
            base_path=dataset_path,
            video_name=f,
            smoothing='2d',
            step=c.getint("data", "step"),
            duration=c.getint("data", "chunks_duration"),
            remove_background=c.getboolean("data", "remove_background")
        ) for f in test_base_file_names]

    for i, tds in enumerate(testing_datasets):
        logger.info(f"Testing dataset {i} contains {len(tds)} samples")

    # class weights
    class_weights = compute_class_weights_puffs(
        dataset,
        w0=args.weight_background,
        w1=args.weight_sparks,
        w2=args.weight_waves,
        w3=args.weight_puffs
    )
    class_weights = torch.tensor(np.float32(class_weights))

    logger.info("Using class weights: {}".format(', '.join(str(w.item()) for w in class_weights)))

    # initialize data loaders
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testing_dataset_loaders = [
        DataLoader(test_dataset, batch_size=args.bach_size, shuffle=False, num_workers=4)
        for test_dataset in testing_datasets
    ]

    # configure unet
    unet_config = unet.UNetConfig(
        steps=c.getint("data", "steps"),
        num_classes=c.getint("network", "num_classes"),
        ndims=c.getint("network", "ndims"),
        border_mode=c.get("network", "border_mode"),
        batch_normalization=c.getboolean("network", "batch_normalization")
    )
    unet_config.feature_map_shapes((c.getint("data", "chunks_duration"), 64, 512))
    network = nn.DataParallel(unet.UNetClassifier(unet_config)).to(device)

    if c.getboolean("general", "wandb_enable"):
        wandb.watch(network)

    if c.getboolean("network", "initialize_weights", fallback="no"):
        network.apply(weights_init)

    # initialize training
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    network.train()
    output_path = os.path.join(c.get("network", "output_relative_path"), args.name)
    logger.info(f"Output directory: {output_path}")
    summary_writer = SummaryWriter(os.path.join(output_path, "summary"), purge_step=0)

    trainer = unet.TrainingManager(
        # training items
        training_step=lambda _: training_step(
            sampler,
            network,
            optimizer,
            device,
            class_weights,
            dataset_loader,
            ignore_frames=c.getint("data", "ignore_frames_loss"),
            ignore_index=c.getint("data", "ignore_index")
        ),
        save_every=5000,
        save_path=output_path,
        managed_objects=unet.managed_objects({
            'network': network,
            'optimizer': optimizer
        }),
        # testing items
        test_function=lambda _: test_function(
            network,
            device,
            class_weights,
            testing_datasets,
            # logger,
            ignore_ind=c.getint("data", "ignore_index")
        ),
        test_every=1000,
        plot_every=1000,
        summary_writer=summary_writer
    )

    # begin training
    if args.load_epoch:
        trainer.load(args.load_epoch)

    test_function(
        network,
        device,
        class_weights,
        testing_datasets,
        # logger,
        ignore_ind=c.getint("data", "ignore_index")
    )

    if args.training:
        logger.info("Starting training")
        trainer.train(args.train_epochs, print_every=100)

    if args.testing:
        logger.info("Starting final validation")
        trainer.run_validation()

