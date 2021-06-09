'''
18.05.2021

Train U-Net using corrected masks based on Miguel's semi-automatic
segmentations, with sparks centres, and cross entropy loss.

The idea is to use this script instead of 20210128_temp_new_masks.py to
perform the same training.

Available videos (saved in folder 'temp_annotation_masks'):
01-11, 15-17, 21-25, 27-28, 33-36 (25 videos)

Of which 01,06,11,22,28 are in the test dataset

'''
import os
import glob
import logging
import argparse

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import unet

from dataset_tools import random_flip, compute_class_weights_puffs, weights_init
from datasets import IDMaskDataset, IDMaskTestDataset
from training_tools import training_step, test_function, sampler, training_step_new
from metrics_tools import take_closest
from options import add_options

from torch.utils.tensorboard import SummaryWriter
import wandb


####################### OTHER PARAMS ###########################################

# TODO: idealmente aggiungere questi all'argument parser nel modello definitivo

# DATASET OPTIONS
dataset_folder = "temp_annotation_masks"

# DATALOADER OPTIONS
n_workers = 1

# TRAINING OPTIONS
test_mode = False # if True it do not run the training
remove_background = True # remove background
criterion = 'nll_loss' # loss function used for training

# step and chunks duration
step = 4
chunks_duration = 64 # power of 2

# configure loss function
ignore_index = 4
ignore_frames_loss = 4

# thresholds for events detection
thresholds = np.arange(0.5, 1, 0.05)
fixed_threshold = 0.95 # t used in plots wrt epochs

################################################################################

if test_mode:
    print("Running in test mode (only execute test function at given epoch)")


# add options
parser = argparse.ArgumentParser(description="Train U-Net")
add_options(parser)
args = parser.parse_args()


wandb.init(project="sparks", name=args.name,
           #notes = 'Cerco di capire il perché del vanishing gradient',
           tags = ['Sparks Project',
                   'Cross entropy loss',
                   'Sparks centres annotated'])


if args.verbose:
    print("Parser arguments: ")
    print("\tVerbose: ",args.verbose)
    print("\tRun name: ",args.name)
    print("\tLoad epoch: ",args.load_epoch)
    print("\tTrain epochs: ",args.train_epochs)
    print("\tBatch size: ",args.batch_size)
    print("\tUsing small dataset: ",args.small_dataset)
    print("\tUsing very small dataset: ",args.very_small_dataset)
    print("\tClass weights coefficients: ",args.weight_background,", ",
                                           args.weight_sparks,", ",
                                           args.weight_waves,", ",
                                           args.weight_puffs)


# configure logger
unet.config_logger("/dev/null")
logger = logging.getLogger(__name__)

# detect device(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
if args.verbose:
    print("Using ", device, "; number of gpus: ", n_gpus)


basepath = os.path.dirname("__file__")


### NETWORK AND DATASETS ###

if args.very_small_dataset:
    dataset_path = os.path.join(basepath,"..","data",dataset_folder,
                                "very_small_dataset")
    if args.verbose:
        print("Train using very small dataset")
else:
    dataset_path = os.path.join(basepath,"..","data",dataset_folder)


dataset = IDMaskDataset(base_path=dataset_path, smoothing='2d',
                          step=step, duration=chunks_duration,
                          remove_background = remove_background)
dataset = unet.TransformedDataset(dataset, random_flip)

if args.verbose:
    print("Samples in training dataset: ", len(dataset))


test_files_names = sorted([".".join(f.split(".")[:-1]) for f in
                   os.listdir(os.path.join(dataset_path,"videos_test"))])
testing_datasets = [IDMaskTestDataset(base_path=dataset_path,
                                    video_name=file, smoothing='2d',
                                    step=step, duration=chunks_duration,
                                    remove_background = remove_background)
                    for file in test_files_names]

if args.verbose:
    print("Samples in each video in testing dataset: ",
          *[len(test_dataset) for test_dataset in testing_datasets])


# compute weights for each class
class_weights = compute_class_weights_puffs(dataset,
                                            w0=args.weight_background,
                                            w1=args.weight_sparks,
                                            w2=args.weight_waves,
                                            w3=args.weight_puffs)
class_weights = torch.tensor(np.float32(class_weights))

if args.verbose:
    print("Class weights:", *(class_weights.tolist()))


# Training dataset loader
dataset_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=n_workers,
                            pin_memory=True)


# List of testing dataset loader
testing_dataset_loaders = [DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=n_workers,
                                      pin_memory=True)
                           for test_dataset in testing_datasets]


unet_config = unet.UNetConfig(
    steps=4,
    num_classes=4,
    ndims=3,
    border_mode='same',
    batch_normalization=False
)

# U-net layers
unet_config.feature_map_shapes((chunks_duration, 64, 512))


network = unet.UNetClassifier(unet_config)
wandb.watch(network)

if n_gpus > 1:
    network = nn.DataParallel(network)
network = network.to(device)

# uncomment to use weights initialization
#network.apply(weights_init)

if criterion == 'nll_loss':
    criterion = nn.NLLLoss(ignore_index=ignore_index,
                           weight=class_weights.to(device))

optimizer = optim.Adam(network.parameters(), lr=1e-4)
network.train();



### PREPARE TRAINING ###

# Compute idx of t in thresholds list that is closest to fixed_threshold
closest_t = take_closest(thresholds, fixed_threshold)
idx_fixed_t = list(thresholds).index(closest_t)

output_path = "epoch_runs/"+args.name
os.makedirs(output_path, exist_ok=True)
summary_writer = SummaryWriter(os.path.join(output_path, "summary"),
                               purge_step=0)
#summary_writer.add_graph(network) # non funziona


trainer = unet.TrainingManager(
    lambda _: training_step_new(network, optimizer, device, criterion,
                                dataset_loader, ignore_frames_loss,
                                summary_writer),
    save_every=5,
    save_path=output_path,
    managed_objects=unet.managed_objects({'network': network,
                                          'optimizer': optimizer}),
    test_function=lambda _: test_function(network, device, criterion,
                                          testing_datasets, logger,
                                          thresholds, idx_fixed_t,
                                          ignore_frames_loss, summary_writer),
    test_every=5,
    plot_every=1,
    summary_writer=summary_writer # comment to use normal plots
)



### TRAIN NETWORK ###
if args.load_epoch:
    trainer.load(args.load_epoch)


# Test network before training
if args.verbose:
    print("Test network before training:")
test_function(network, device, criterion, testing_datasets, logger, thresholds,
              idx_fixed_t, ignore_frames_loss, summary_writer)

if test_mode:
    exit()


# Train network
trainer.train(args.train_epochs, print_every=1)

summary_writer.close()
