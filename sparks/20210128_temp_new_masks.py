'''
28.01.2021

Train U-Net using corrected masks based on Miguel's semi-automatic
segmentations.

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
from training_tools import training_step, test_function, sampler
from options import add_options

from tensorboardX import SummaryWriter
import wandb


dataset_folder = "temp_annotation_masks"

test_mode = True

# sparks in the masks have already the correct shape (radius and ignore index)

# ignored index
ignore_index = 4

# event size
#radius_event = 3.5

# remove background
remove_background = True

# step and chunks duration
step = 4
chunks_duration = 16 # power of 2
ignore_frames_loss = (chunks_duration-step)//2 # frames ignored by loss fct

# add options
parser = argparse.ArgumentParser(description="Train U-Net")
add_options(parser)
args = parser.parse_args()


wandb.init(project="sparks", name=args.name)


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


unet.config_logger("/dev/null")
logger = logging.getLogger(__name__)


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
                          #radius_event = radius_event,
                          remove_background = remove_background)
dataset = unet.TransformedDataset(dataset, random_flip)

if args.verbose:
    print("Samples in training dataset: ", len(dataset))


test_files_names = sorted([".".join(f.split(".")[:-1]) for f in
                   os.listdir(os.path.join(dataset_path,"videos_test"))])
testing_datasets = [IDMaskTestDataset(base_path=dataset_path,
                                    video_name=file, smoothing='2d',
                                    step=step, duration=chunks_duration,
                                    #radius_event = radius_event,
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
                            shuffle=True, num_workers=4)


# List of testing dataset loader
testing_dataset_loaders = [DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4)
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

optimizer = optim.Adam(network.parameters(), lr=1e-4)
network.train();



### PREPARE TRAINING ###

output_path = "runs/"+args.name
os.makedirs(output_path, exist_ok=True)
summary_writer = SummaryWriter(os.path.join(output_path, "summary"),
                               purge_step=0)


trainer = unet.TrainingManager(
    lambda _: training_step(sampler, network, optimizer, device, class_weights,
                            dataset_loader, ignore_frames=ignore_frames_loss,
                            ignore_ind=ignore_index),
    save_every=5000,
    save_path=output_path,
    managed_objects=unet.managed_objects({'network': network,
                                          'optimizer': optimizer}),
    test_function=lambda _: test_function(network,device,class_weights,
                                          testing_datasets,logger,
                                          ignore_ind=ignore_index),
    test_every=1000,
    plot_every=1000,
    summary_writer=summary_writer # comment to use normal plots
)



### TRAIN NETWORK ###
if args.load_epoch:
    trainer.load(args.load_epoch)


# Test network before training
if args.verbose:
    print("Test network before training:")
test_function(network,device,class_weights,testing_datasets,logger,
              ignore_ind=ignore_index)

if test_mode:
    exit()


# Train network
trainer.train(args.train_epochs, print_every=100)
