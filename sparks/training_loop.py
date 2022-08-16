#!/usr/bin/env python
# coding: utf-8

# 15.08.2022
#
# ### **GOAL**: training basato sul dataset usando l'architettura della UNet trovata su GitHub (https://github.com/ELEKTRONN/elektronn3).
#
# Uso questo notebook per creare il codice necessario a fare il training. Quando avrò finito, se funziona, lo copierò in uno script classico di Python (.py).

import sys
import os
import logging
import argparse
import configparser
import glob


# In[2]:


import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb


# In[3]:


#import unet
from dataset_tools import random_flip, random_flip_noise, compute_class_weights, weights_init
from datasets import SparkDataset
from training_tools import training_step, test_function, sampler
from metrics_tools import take_closest
from other_losses import FocalLoss, LovaszSoftmax3d, SumFocalLovasz


# In[4]:


import unet


# In[5]:


from new_unet import UNet


# In[6]:


BASEDIR = os.path.abspath('')
logger = logging.getLogger(__name__)


# In[7]:


#parser = argparse.ArgumentParser("Spark & Puff detector using U-Net (ELEKTRONN3 model).")

############################# load config file #############################

#parser.add_argument(
#    'config',
#    type=str,
#    help="Input config file, used to configure training"
#)
#args = parser.parse_args()

config_directory = "config_files"
#CONFIG_FILE = os.path.join(BASEDIR, "config_files", args.config)
CONFIG_FILE = os.path.join(BASEDIR,
                           "config_files",
                           "config_test_new_unet_architecture.ini")
c = configparser.ConfigParser()
if os.path.isfile(CONFIG_FILE):
    logger.info(f"Loading {CONFIG_FILE}")
    c.read(CONFIG_FILE)
else:
    logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")


# In[8]:


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
if (params['loss_function'] == 'focal_loss') or( params['loss_function'] == "sum_losses"):
    params['gamma'] = c.getfloat("training", "gamma", fallback=2.0)
if params['loss_function'] == 'sum_losses':
    params['w'] = c.getfloat("training", "w", fallback=0.5)
params['lr_start'] = c.getfloat("training", "lr_start", fallback=1e-4)

# data params
params['dataset_basedir'] = c.get("data", "relative_path")
params['dataset_size'] = c.get("data", "size", fallback="full")
params['batch_size'] = c.getint("general", "batch_size", fallback="1")
params['data_duration'] = c.getint("data", "chunks_duration")
params['data_step'] = c.getint("data", "step")
params['ignore_frames_loss'] = c.getint("data", "ignore_frames_loss")
params['data_smoothing'] = c.get("data", "smoothing", fallback="2d")
params['norm_video'] = c.get("data", "norm_video", fallback="chunk")
params['remove_background'] = c.get("data", "remove_background", fallback='average')
params['only_sparks'] = c.getboolean("data", "only_sparks", fallback=False)
params['noise_data_augmentation'] = c.getboolean("data", "noise_data_augmentation", fallback=False)
params['sparks_type'] = c.get("data", "sparks_type", fallback="peaks")

# UNet params
params['unet_steps'] = c.getint("network", "step")
params['first_layer_channels'] = c.getint("network", "first_layer_channels")
params['temporal_reduction'] = c.getboolean("network", "temporal_reduction", fallback=False)
params['num_channels'] = c.getint("network", "num_channels", fallback=1)

# Testing params
params['t_detection_sparks'] = c.getfloat("testing", "t_sparks")
params['t_detection_puffs'] = c.getfloat("testing", "t_puffs")
params['t_detection_waves'] = c.getfloat("testing", "t_waves")
params['sparks_min_radius'] = c.getint("testing", "sparks_min_radius")
params['puffs_min_radius'] = c.getint("testing", "puffs_min_radius")
params['waves_min_radius'] = c.getint("testing", "waves_min_radius")


# In[9]:


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


# In[10]:


############################# configure wandb ##############################

if c.getboolean("general", "wandb_enable", fallback=False):
    wandb.init(project=c.get("general", "wandb_project_name"), name=params['name'])
    logging.getLogger('wandb').setLevel(logging.DEBUG)
    #wandb.save(CONFIG_FILE)


# In[11]:


############################# print parameters #############################

logger.info("Command parameters:")
for k, v in params.items():
    logger.info(f"{k:>18s}: {v}")
    # TODO: AGGIUNGERE TUTTI I PARAMS NECESSARI DA PRINTARE


# In[12]:


############################ init random seeds #############################

torch.manual_seed(0)
#random.seed(0)
np.random.seed(0)


# In[13]:


########################### detect CUDA devices ############################
if c.getboolean("general", "cuda", fallback=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = True
else:
    device = 'cpu'
    pin_memory = False
n_gpus = torch.cuda.device_count()
logger.info(f"Using torch device {device}, with {n_gpus} GPUs")


# In[14]:


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


# In[15]:


# initialize training dataset
dataset_path = os.path.realpath(f"{BASEDIR}/{params['dataset_basedir']}")
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
    sparks_type=params['sparks_type']
)


# In[16]:


# modified since not importing Pablo's code for UNet
from torch.utils.data import Dataset
class TransformedDataset(Dataset):

    def __init__(self, source_dataset, transform):
        self.source_dataset = source_dataset
        self.transform = transform

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):

        value = self.source_dataset[idx]

        if isinstance(value, tuple):
            return self.transform(*value)

        return self.transform(value)


# In[17]:


# modified since not importing Pablo's code for UNet
# apply transforms
if params['noise_data_augmentation']:
    dataset = TransformedDataset(dataset, random_flip_noise)
else:
    dataset = TransformedDataset(dataset, random_flip)

logger.info(f"Samples in training dataset: {len(dataset)}")


# In[18]:


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
            ignore_frames=params['ignore_frames_loss']
            ) for sample_id in test_sample_ids]

for i, tds in enumerate(testing_datasets):
    logger.info(f"Testing dataset {i} contains {len(tds)} samples")


# In[19]:


# class weights
class_weights = compute_class_weights(dataset)
logger.info("Using class weights: {}".format(', '.join(str(w.item()) for w in class_weights)))


# In[20]:


# initialize data loaders
dataset_loader = DataLoader(dataset,
                            batch_size=params['batch_size'],
                            shuffle=True,
                            num_workers=c.getint("training", "num_workers"),
                            pin_memory=pin_memory)
testing_dataset_loaders = [
    DataLoader(test_dataset,
               batch_size=params['batch_size'],
               shuffle=False,
               num_workers=c.getint("training", "num_workers"))
    for test_dataset in testing_datasets
]


# In[21]:


############################## configure UNet ##############################

#unet_config = unet.UNetConfig(
#    steps=params['unet_steps'],
#    first_layer_channels=params['first_layer_channels'],
#    num_classes=c.getint("network", "num_classes"),
#    ndims=c.getint("network", "ndims"),
#    dilation=c.getint("network", "dilation", fallback=1),
#    border_mode=c.get("network", "border_mode"),
#    batch_normalization=c.getboolean("network", "batch_normalization"),
#    num_input_channels=params['num_channels'],
#)

#if not params['temporal_reduction']:
#    network = unet.UNetClassifier(unet_config)
#else:
#    assert params['data_duration'] % params['num_channels'] == 0, \
#    "using temporal reduction chunks_duration must be a multiple of num_channels"
#    network = TempRedUNet(unet_config)


# In[74]:


from new_unet import UNet


# In[75]:


############################## configure UNet ##############################
out_channels = c.getint("network", "num_classes")
network = UNet(
    in_channels=params['num_channels'],
    out_channels=out_channels,
    n_blocks=params['unet_steps'],
    start_filts=params['first_layer_channels'],
    #up_mode = ... # TESTARE DIVERSE POSSIBILTÀ, e.g.'resizeconv_nearest' to avoid checkerboard artifacts
    merge_mode='concat', # Default, dicono che funziona meglio
    #planar_blocks=(0,), # magari capire cos'è e testarlo ??
    activation='relu',
    normalization='batch', # Penso che nell'implementazione di Pablo è 'none'
    attention=False, # magari da testare con 'True' ??
    #full_norm=False,  # Uncomment to restore old sparse normalization scheme
    dim=c.getint("network", "ndims"),
    #conv_mode='valid',  # magari testare, ha dei vantaggi a quanto pare...
    #up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
).to(device)

# DOC: https://elektronn3.readthedocs.io/en/latest/source/elektronn3.models.unet.html


# In[76]:


if device != "cpu":
    network = nn.DataParallel(network).to(device)
    torch.backends.cudnn.benchmark = True

if c.getboolean("general", "wandb_enable"):
    wandb.watch(network)

if c.getboolean("network", "initialize_weights", fallback=False):
    logger.info("Initializing UNet weights...")
    network.apply(weights_init)


# In[77]:


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


# In[78]:


########################### initialize training ############################

optimizer = optim.Adam(network.parameters(), lr=params['lr_start'])
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
                          alpha=class_weights,
                          gamma=params['gamma'])
elif params['loss_function'] == 'lovasz_softmax':
    criterion = LovaszSoftmax3d(classes='present',
                                per_image=False,
                                ignore=c.getint("data", "ignore_index"))
elif params['loss_function'] == 'sum_losses':
    criterion = SumFocalLovasz(classes ='present',
                               per_image = False,
                               ignore = c.getint("data", "ignore_index"),
                               alpha = class_weights,
                               gamma = params['gamma'],
                               reduction = 'mean',
                               w = params['w'])


# In[86]:


from training_tools import test_function


# In[87]:


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
        #network=network,
        network=network,
        device=device,
        criterion=criterion,
        testing_datasets=testing_datasets,
        logger=logger,
        ignore_frames=params['ignore_frames_loss'],
        wandb_log=c.getboolean("general", "wandb_enable", fallback=False),
        training_name=c.get("general", "run_name"),
        training_mode=True
    ),
    test_every=c.getint("training", "test_every", fallback=1000),
    plot_every=c.getint("training", "plot_every", fallback=1000),
    summary_writer=summary_writer
)


# In[81]:


############################## start training ##############################

if params['load_epoch'] != 0:
    trainer.load(params['load_epoch'])


# In[ ]:


if params['training']:
    logger.info("Validate network before training")
    trainer.run_validation()
    logger.info("Starting training")
    trainer.train(params['train_epochs'], print_every=100)


# In[88]:


if params['testing']:
    logger.info("Starting final validation")
    trainer.run_validation()