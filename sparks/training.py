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
from torch.utils.tensorboard import SummaryWriter
import wandb

import unet
from new_unet import UNet

from training_inference_tools import (random_flip,
                                      random_flip_noise,
                                      compute_class_weights,
                                      weights_init,
                                      training_step,
                                      test_function,
                                      sampler)
from datasets import SparkDataset
from custom_losses import FocalLoss, LovaszSoftmax3d, SumFocalLovasz
from architectures import TempRedUNet

BASEDIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ############################# fixed parameters #############################

    # General params
    verbosity = 3
    logfile = None # change this when publishing finished project on github
    wandb_project_name = 'sparks'
    output_relative_path = 'runs/' # directory where output, saved params and
                                   # testing results are saved

    # Dataset parameters
    ignore_index = 4 # label ignored during training
    num_classes = 4 # i.e., BG, sparks, waves, puffs
    ndims = 3 # using 3D data


    ############################# load config file #############################

    parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")
    parser.add_argument(
        'config',
        type=str,
        help="Input config file, used to configure training"
    )
    args = parser.parse_args()

    CONFIG_FILE = os.path.join(BASEDIR, "config_files", args.config)
    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")

    ############################## set parameters ##############################

    params = {}

    # training params
    params['run_name'] = c.get("training", "run_name", fallback="TEST") # Run name
    params['load_run_name'] = c.get("training", "load_run_name", fallback=None)
    params['load_epoch'] = c.getint("training", "load_epoch", fallback=0)
    params['train_epochs'] = c.getint("training", "train_epochs", fallback=5000)
    params['criterion'] = c.get("training", "criterion", fallback="nll_loss")
    params['lr_start'] = c.getfloat("training", "lr_start", fallback=1e-4)
    params['ignore_frames_loss'] = c.getint("training", "ignore_frames_loss")
    if (params['criterion'] == 'focal_loss') or( params['criterion'] == "sum_losses"):
        params['gamma'] = c.getfloat("training", "gamma", fallback=2.0)
    if params['criterion'] == 'sum_losses':
        params['w'] = c.getfloat("training", "w", fallback=0.5)
    params['cuda'] = c.getboolean("training", "cuda")

    # dataset params
    params['relative_path'] = c.get("dataset", "relative_path")
    params['dataset_size'] = c.get("dataset", "dataset_size", fallback="full")
    params['batch_size'] = c.getint("dataset", "batch_size", fallback=1)
    params['num_workers'] = c.getint("dataset", "num_workers", fallback=1)
    params['data_duration'] = c.getint("dataset", "data_duration")
    params['data_step'] = c.getint("dataset", "data_step")
    params['data_smoothing'] = c.get("dataset", "data_smoothing", fallback="2d")
    params['norm_video'] = c.get("dataset", "norm_video", fallback="chunk")
    params['remove_background'] = c.get("dataset", "remove_background", fallback='average')
    params['only_sparks'] = c.getboolean("dataset", "only_sparks", fallback=False)
    params['noise_data_augmentation'] = c.getboolean("dataset", "noise_data_augmentation", fallback=False)
    params['sparks_type'] = c.get("dataset", "sparks_type", fallback="peaks")
    params['inference'] = c.get("dataset", "inference", fallback="overlap")

    # UNet params
    params['nn_architecture'] = c.get("network", "nn_architecture", fallback='pablos_unet')
    params['unet_steps'] = c.getint("network", "unet_steps")
    params['first_layer_channels'] = c.getint("network", "first_layer_channels")
    params['num_channels'] = c.getint("network", "num_channels", fallback=1)
    params['dilation'] = c.getboolean("network", "dilation", fallback=1)
    params['border_mode'] = c.get("network", "border_mode")
    params['batch_normalization'] = c.get("network", "batch_normalization", fallback='none')
    params['temporal_reduction'] = c.getboolean("network", "temporal_reduction", fallback=False)
    params['initialize_weights'] = c.getboolean("network", "initialize_weights", fallback=False)
    if params['nn_architecture'] == 'github_unet':
        params['attention'] = c.getboolean("network", 'attention')
        params['up_mode'] = c.get("network", 'up_mode')

    ############################# configure logger #############################

    level_map = {3: logging.DEBUG, 2: logging.INFO, 1: logging.WARNING, 0: logging.ERROR}
    log_level = level_map[verbosity]
    log_handlers = (logging.StreamHandler(sys.stdout), )

    # use this when project is finished:
    #if logfile:
    #    if not os.path.isdir(os.path.basename(logfile)):
    #        logger.info("Creating parent directory for logs")
    #        os.mkdir(os.path.basename(logfile))
    #
    #    if os.path.isdir(logfile):
    #        logfile_path = os.path.abspath(os.path.join(logfile, f"{__name__}.log"))
    #    else:
    #        logfile_path = os.path.abspath(logfile)
    #
    #    logger.info(f"Storing logs in {logfile_path}")
    #    file_handler = logging.RotatingFileHandler(
    #        filename=logfile_path,
    #        maxBytes=(1024 * 1024 * 8),  # 8 MB
    #        backupCount=4,
    #    )
    #    log_handlers += (file_handler, )

    logging.basicConfig(
        level=log_level,
        format='[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}',
        style='{',
        datefmt="%H:%M:%S",
        handlers=log_handlers)

    ############################# configure wandb ##############################

    if c.getboolean("general", "wandb_enable", fallback=False):
        wandb.init(project=wandb_project_name,
                   name=params['run_name'],
                   notes=c.get("general", "wandb_notes", fallback=None))
        logging.getLogger('wandb').setLevel(logging.DEBUG)
        #wandb.save(CONFIG_FILE)

    ############################# print parameters #############################

    logger.info("Command parameters:")
    for k, v in params.items():
        logger.info(f"{k:>18s}: {v}")
        # TODO: AGGIUNGERE TUTTI I PARAMS NECESSARI DA PRINTARE

    ############################ init random seeds #############################

    #torch.manual_seed(0)
    #random.seed(0)
    #np.random.seed(0)

    ############################ configure datasets ############################

    # select samples that are used for training and testing
    if params['dataset_size'] == 'full':
        train_sample_ids = ["01","02","03","04","06","07","08","09",
                            "11","12","13","14","16","17","18","19",
                            "21","22","23","24","27","28","29",
                            "30","33","35","36","38","39",
                            "41","42","43","44","46"]
        test_sample_ids = ["05","10","15","20","25","32","34","40","45"]
    elif params['dataset_size'] == 'minimal':
        train_sample_ids = ["01"]
        test_sample_ids = ["34"]
    else:
        logger.error(f"{params['dataset_size']} is not a valid dataset size.")
        exit()

    # detect CUDA devices
    if params['cuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pin_memory = True
    else:
        device = 'cpu'
        pin_memory = False
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    # set if temporal reduction is used
    if params['temporal_reduction']:
        logger.info(f"Using temporal reduction with {params['num_channels']} channels")

    # normalize whole videos or chunks individually
    if params['norm_video'] == 'chunk':
        logger.info("Normalizing each chunk using min and max")
    elif params['norm_video'] == 'movie':
        logger.info("Normalizing whole video using min and max")
    elif params['norm_video'] == 'abs_max':
        logger.info("Normalizing whole video using 16-bit absolute max")

    # initialize training dataset
    dataset_path = os.path.realpath(f"{BASEDIR}/{params['relative_path']}")
    assert os.path.isdir(dataset_path), f"\"{dataset_path}\" is not a directory"
    logger.info(f"Using {dataset_path} as dataset root path")
    dataset = SparkDataset(
        base_path=dataset_path,
        sample_ids=train_sample_ids,
        testing=False,
        smoothing=params['data_smoothing'],
        step=params['data_step'],
        duration=params['data_duration'],
        remove_background=params['remove_background'],
        temporal_reduction=params['temporal_reduction'],
        num_channels=params['num_channels'],
        normalize_video=params['norm_video'],
        only_sparks=params['only_sparks'],
        sparks_type=params['sparks_type'],
        ignore_index=ignore_index,
        inference=None
    )

    # apply transforms
    if params['noise_data_augmentation']:
        dataset = unet.TransformedDataset(dataset, random_flip_noise)
    else:
        dataset = unet.TransformedDataset(dataset, random_flip)

    logger.info(f"Samples in training dataset: {len(dataset)}")

    # initialize testing dataset
    pattern_test_filenames = os.path.join(f"{dataset_path}","videos_test",
                                           "[0-9][0-9]_video.tif")

    testing_datasets = [
                SparkDataset(
                base_path=dataset_path,
                sample_ids=[sample_id],
                testing=True,
                smoothing=params['data_smoothing'],
                step=params['data_step'],
                duration=params['data_duration'],
                remove_background=params['remove_background'],
                temporal_reduction=params['temporal_reduction'],
                num_channels=params['num_channels'],
                normalize_video=params['norm_video'],
                only_sparks=params['only_sparks'],
                sparks_type=params['sparks_type'],
                ignore_frames=params['ignore_frames_loss'],
                ignore_index=ignore_index,
                inference=params['inference']
            ) for sample_id in test_sample_ids]

    for i, tds in enumerate(testing_datasets):
        logger.info(f"Testing dataset {i} contains {len(tds)} samples")

    # initialize data loaders
    dataset_loader = DataLoader(dataset,
                                batch_size=params['batch_size'],
                                shuffle=True,
                                num_workers=params['num_workers'],
                                pin_memory=pin_memory)
    #testing_dataset_loaders = [
    #    DataLoader(test_dataset,
    #               batch_size=params['batch_size'],
    #               shuffle=False,
    #               num_workers=params['num_workers'])
    #    for test_dataset in testing_datasets
    #] NON VIENE USATO ???

    ############################## configure UNet ##############################

    if params['nn_architecture'] == 'pablos_unet':

        batch_norm = {'batch': True, 'none': False}

        unet_config = unet.UNetConfig(
            steps=params['unet_steps'],
            first_layer_channels=params['first_layer_channels'],
            num_classes=num_classes,
            ndims=ndims,
            dilation=params['dilation'],
            border_mode=params['border_mode'],
            batch_normalization=batch_norm[params['batch_normalization']],
            num_input_channels=params['num_channels'],
        )

        if not params['temporal_reduction']:
            network = unet.UNetClassifier(unet_config)
        else:
            assert params['data_duration'] % params['num_channels'] == 0, \
            "using temporal reduction chunks_duration must be a multiple of num_channels"
            network = TempRedUNet(unet_config)

    elif params['nn_architecture'] == 'github_unet':
        network = UNet(
            in_channels=params['num_channels'],
            out_channels=num_classes,
            n_blocks=params['unet_steps']+1,
            start_filts=params['first_layer_channels'],
            #up_mode = 'transpose', # TESTARE DIVERSE POSSIBILTÀ
            #up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
            merge_mode='concat', # Default, dicono che funziona meglio
            #planar_blocks=(0,), # magari capire cos'è e testarlo ??
            activation='relu',
            normalization=params['batch_normalization'], # Penso che nell'implementazione di Pablo è 'none'
            attention=params['attention'], # magari da testare con 'True' ??
            #full_norm=False,  # Uncomment to restore old sparse normalization scheme
            dim=ndims,
            conv_mode=params['border_mode'],  # 'valid' ha dei vantaggi a quanto pare...
        )

    if device != "cpu":
        network = nn.DataParallel(network).to(device)
        torch.backends.cudnn.benchmark = True

    if c.getboolean("general", "wandb_enable", fallback=False):
        wandb.watch(network)

    if params['initialize_weights']:
        logger.info("Initializing UNet weights...")
        network.apply(weights_init)

    ########################### initialize training ############################

    optimizer = optim.Adam(network.parameters(), lr=params['lr_start'])
    network.train()

    output_path = os.path.join(output_relative_path, params['run_name'])
    logger.info(f"Output directory: {output_path}")

    summary_writer = SummaryWriter(os.path.join(output_path, "summary"),
                                   purge_step=0)

    if params['load_run_name'] != None:
        load_path = os.path.join(output_relative_path, params['load_run_name'])
        logger.info(f"Model loaded from directory: {load_path}")
    else:
        load_path = None

    # class weights
    if params['criterion'] in ['nll_loss', 'focal_loss', 'sum_losses']:
        class_weights = compute_class_weights(dataset)
        logger.info("Using class weights: {}".format(', '.join(str(w.item()) for w in class_weights)))

    if params['criterion'] == 'nll_loss':
        criterion = nn.NLLLoss(ignore_index=ignore_index,
                               weight=class_weights.to(device))
    elif params['criterion'] == 'focal_loss':
        criterion = FocalLoss(reduction='mean',
                              ignore_index=ignore_index,
                              alpha=class_weights,
                              gamma=params['gamma'])
    elif params['criterion'] == 'lovasz_softmax':
        criterion = LovaszSoftmax3d(classes='present',
                                    per_image=False,
                                    ignore=ignore_index)
    elif params['criterion'] == 'sum_losses':
        criterion = SumFocalLovasz(classes='present',
                                   per_image=False,
                                   ignore=ignore_index,
                                   alpha=class_weights,
                                   gamma=params['gamma'],
                                   reduction='mean',
                                   w=params['w'])

    # directory where predicted class movies are saved
    preds_output_dir = os.path.join(output_relative_path,
                                    params['run_name'],
                                    'predictions')
    os.makedirs(preds_output_dir, exist_ok=True)

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
        #load_path=load_path,
        save_path=output_path,
        managed_objects=unet.managed_objects({
            'network': network,
            'optimizer': optimizer
        }),
        # testing items
        test_function=lambda _: test_function(
            network=network,
            device=device,
            criterion=criterion,
            testing_datasets=testing_datasets,
            logger=logger,
            ignore_frames=params['ignore_frames_loss'],
            wandb_log=c.getboolean("general", "wandb_enable", fallback=False),
            training_name=params['run_name'],
            output_dir=preds_output_dir,
            training_mode=True
        ),
        test_every=c.getint("training", "test_every", fallback=1000),
        plot_every=c.getint("training", "test_every", fallback=1000),
        summary_writer=summary_writer
    )

    ############################## start training ##############################

    if params['load_epoch'] != 0:
        trainer.load(params['load_epoch'])

    if c.getboolean("general", "training"): # Run training procedure on data
        logger.info("Validate network before training")
        trainer.run_validation()
        logger.info("Starting training")
        trainer.train(params['train_epochs'],
                      print_every=c.getint("training", "print_every", fallback=100))

    if c.getboolean("general", "testing"): # Run training procedure on data
        logger.info("Starting final validation")
        trainer.run_validation()
