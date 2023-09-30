"""
Create classes to manage the project.

Classes:
    ProjectConfig: stores all the global variables for the project.
    TrainingConfig: loads settings from a configuration file, initializes
                    parameters, and configures WandB (Weights and Biases)
                    logging.

Author: Prisca Dotti
Last modified: 28.09.2023
"""


import argparse
from configparser import ConfigParser
import logging
import math
import os
import sys
import numpy as np
import torch

import wandb

logger = logging.getLogger(__name__)


class ProjectConfig:
    def __init__(self):
        """
        Initialize the configuration object.
        The configuration object stores all the global variables for the project.
        """
        # Get basedir of the project
        self.basedir = os.path.dirname(os.path.realpath(__file__))

        ### General parameters ###

        self.logfile = None  # Change this when publishing the finished project on GitHub
        self.verbosity = 2
        self.debug_mode = False
        # wandb_project_name = "TEST"
        self.wandb_project_name = "sparks2"
        # Directory where output, saved parameters, and testing results are saved
        self.output_relative_path = "runs"

        ### Dataset parameters ###

        self.ndims = 3  # Using 3D data
        self.pixel_size = 0.2  # 1 pixel = 0.2 um x 0.2 um
        self.time_frame = 6.8  # 1 frame = 6.8 ms

        self.classes_dict = {
            "background": 0,
            "sparks": 1,
            "waves": 2,
            "puffs": 3
        }

        self.num_classes = len(self.classes_dict)
        self.ignore_index = self.num_classes + 1  # Label ignored during training

        # Include ingore index in the classes dictionary
        self.classes_dict["ignore"] = self.ignore_index

        ### Physiological parameters ###

        ## Sparks (1) parameters ##

        # To get sparks locations
        # Minimum distance in space between sparks
        self.min_dist_xy = round(1.8 / self.pixel_size)
        # Minimum distance in time between sparks
        self.min_dist_t = round(20 / self.time_frame)
        # Connectivity mask of sparks
        self._radius = math.ceil(self.min_dist_xy / 2)
        self._y, self._x = np.ogrid[-self._radius: self._radius + 1,
                                    -self._radius: self._radius + 1]
        self._disk = self._x**2 + self._y**2 <= self._radius**2
        self.conn_mask = np.stack([self._disk] * self.min_dist_t, axis=0)
        # Sigma value used for sample smoothing in sparks peaks detection
        self.sparks_sigma = 3

        # To remove small sparks after detection
        self.spark_min_width = 3
        self.spark_min_t = 3

        ## Waves (2) parameters ##

        # To remove small waves after detection
        self.wave_min_width = round(15 / self.pixel_size)

        ## Puffs (3) parameters ##

        # To remove small puffs after detection
        self.puff_min_t = 5

        ### Events detection parameters ###

        # Connectivity for event instances detection
        self.connectivity = 26
        # Maximal gap between two predicted puffs or waves that belong together
        self.max_gap = 2  # i.e., 2 empty frames

        # Parameters for correspondence computation
        # (threshold for considering annotated and pred ROIs a match)
        self.iomin_t = 0.5


# Initialize the configuration object
config = ProjectConfig()


class TrainingConfig:
    def __init__(self, training_config_file):
        """
        Initialize the training configuration object.
        A configuration manager for loading settings from a configuration file,
        initializing parameters, and configuring WandB (Weights and Biases)
        logging, if wandb_project_name is not None.

        Parameters:
        training_config_file : str
            Path to the training configuration file.

        Attributes:
        TODO: add explanation of all attributes...

        params.norm_video:
            "chunk": Normalizing each chunk using min and max
            "movie": Normalizing whole video using min and max
            "abs_max": "Normalizing whole video using 16-bit absolute max"

        params.nn_architecture:
            "pablos_unet": classical 3D U-Net
            "github_unet": other 3D U-Net implementation from GitHub that has
                more options, such as batch normalization, attention, etc.
            "openai_unet": 3D U-Net implementation from OpenAI that has more
                options, such as residual blocks, attention, etc.

        params.temporal_reduction:
            If True, apply convolutional layers with stride 2 to reduce the
            temporal dimension prior to the U-Net (using TempRedUNet). Tried
            this to fit longer sequences in memory, but it didn't help much.

        params.num_channels:
            >0 if using temporal_reduction, value depends on temporal
            reduction configuration

        params.data_step:
            Step between two chunks extracted from the sample

        params.data_duration:
            Duration of a chunk

        params.data_smoothing:
            If '2d' or '3d', preprocess movie with simple convolution
            (probably useless)

        params.remove_background:
            If 'moving' or 'average', remove background from input movies
            accordingly

        params.only_sparks:
            If True, train using only sparks annotations

        params.sparks_type:
            Can be 'raw', 'peaks' (use smaller annotated ROIs), or 'dilated'
            (add ignore regions around events)

        params.ignore_frames_loss:
            If testing, used to ignore events in first and last frames

        """

        # Configure logging
        self.configure_logging()

        # Load configuration file
        self.training_config_file = training_config_file
        self.load_configuration_file()

        # Load configuration parameters here...
        self.load_training_params()
        self.load_dataset_params()
        self.load_unet_params()

        # Configure WandB
        self.configure_wandb()

        # Set the device to use for training
        self.set_device()

    def configure_logging(self):
        # Define a mapping of verbosity levels
        level_map = {
            3: logging.DEBUG,
            2: logging.INFO,
            1: logging.WARNING,
            0: logging.ERROR,
        }

        # Get the log level based on the configured verbosity
        log_level = level_map[config.verbosity]

        # Create a list of log handlers starting with stdout
        log_handlers = (logging.StreamHandler(sys.stdout),)

        # Configure file logging if a logfile is provided
        # (use this when project is finished)
        if config.logfile:
            self.configure_file_logging(log_handlers)

        # Configure the basic logging settings
        logging.basicConfig(
            level=log_level,
            format="[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}",
            style="{",
            datefmt="%H:%M:%S",
            handlers=log_handlers,
        )

    def configure_file_logging(self, log_handlers):
        # Add file logging handler if a logfile is provided
        log_dir = os.path.basename(config.logfile)
        if not os.path.isdir(log_dir):
            logger.info("Creating parent directory for logs")
            os.mkdir(log_dir)

        if os.path.isdir(config.logfile):
            logfile_path = os.path.abspath(
                os.path.join(config.logfile, f"{__name__}.log"))
        else:
            logfile_path = os.path.abspath(config.logfile)

        logger.info(f"Storing logs in {logfile_path}")
        file_handler = logging.RotatingFileHandler(
            filename=logfile_path,
            maxBytes=(1024 * 1024 * 8),  # 8 MB
            backupCount=4,
        )
        log_handlers += (file_handler, )

    def load_configuration_file(self):
        # Initialize the ConfigParser
        self.c = ConfigParser()

        # Read the configuration file if it exists
        if os.path.isfile(self.training_config_file):
            logger.info(f"Loading {self.training_config_file}")
            self.c.read(self.training_config_file)
        else:
            logger.warning(
                f"No config file found at {self.training_config_file}, trying to use fallback values."
            )

    def load_training_params(self):
        # Load training parameters
        training_section = self.c["training"]

        self.run_name = training_section.get(
            "run_name", fallback="TEST")
        self.load_run_name = training_section.get(
            "load_run_name", fallback=None)
        self.load_epoch = training_section.getint("load_epoch", fallback=0)
        self.train_epochs = training_section.getint(
            "train_epochs", fallback=5000)
        self.criterion = training_section.get(
            "criterion", fallback="nll_loss")
        self.lr_start = training_section.getfloat("lr_start", fallback=1e-4)
        self.ignore_frames_loss = training_section.getint(
            "ignore_frames_loss", fallback=0)
        if (self.criterion == "focal_loss") or (self.criterion == "sum_losses"):
            self.gamma = training_section.getfloat("gamma", fallback=2.0)
        if self.criterion == "sum_losses":
            self.w = training_section.getfloat("w", fallback=0.5)
        self.cuda = training_section.getboolean("cuda")
        self.scheduler = training_section.get("scheduler", fallback=None)
        if self.scheduler == "step":
            self.scheduler_step_size = training_section.getint("step_size")
            self.scheduler_gamma = training_section.getfloat("gamma")
        self.optimizer = training_section.get("optimizer", fallback="adam")

    def load_dataset_params(self):
        # Load dataset parameters
        dataset_section = self.c["dataset"]

        self.relative_path = dataset_section.get("relative_path")
        self.dataset_size = dataset_section.get(
            "dataset_size", fallback="full")
        self.batch_size = dataset_section.getint("batch_size", fallback=1)
        # dataset_section.getint("num_workers", fallback=1)
        self.num_workers = 0
        self.data_duration = dataset_section.getint("data_duration")
        self.data_step = dataset_section.getint("data_step", fallback=1)
        self.testing_data_step = self.c.getint("testing", "data_step",
                                               fallback=self.data_step)
        self.data_smoothing = dataset_section.get(
            "data_smoothing", fallback="2d")
        self.norm_video = dataset_section.get(
            "norm_video", fallback="chunk")
        self.remove_background = dataset_section.get(
            "remove_background", fallback="average")
        self.only_sparks = dataset_section.getboolean(
            "only_sparks", fallback=False)
        self.noise_data_augmentation = dataset_section.getboolean(
            "noise_data_augmentation", fallback=False)
        self.sparks_type = dataset_section.get("sparks_type", fallback="peaks")
        self.inference = dataset_section.get("inference", fallback="overlap")

    def load_unet_params(self):
        # Load UNet parameters
        network_section = self.c["network"]

        self.nn_architecture = network_section.get(
            "nn_architecture", fallback="pablos_unet"
        )
        if self.nn_architecture == "unet_lstm":
            self.bidirectional = network_section.getboolean("bidirectional")
        self.unet_steps = network_section.getint("unet_steps")
        self.first_layer_channels = network_section.getint(
            "first_layer_channels")
        self.num_channels = network_section.getint("num_channels", fallback=1)
        self.dilation = network_section.getboolean("dilation", fallback=1)
        self.border_mode = network_section.get("border_mode")
        self.batch_normalization = network_section.get(
            "batch_normalization", fallback="none"
        )
        self.temporal_reduction = network_section.getboolean(
            "temporal_reduction", fallback=False
        )
        self.initialize_weights = network_section.getboolean(
            "initialize_weights", fallback=False
        )
        if self.nn_architecture == "github_unet":
            self.attention = network_section.getboolean("attention")
            self.up_mode = network_section.get("up_mode")
        if self.nn_architecture == "openai_unet":
            self.num_res_blocks = network_section.getint("num_res_blocks")

    def initialize_wandb(self):
        # Only resume when loading the same saved model
        if self.load_epoch > 0 and self.load_run_name is None:
            resume = "must"
        else:
            resume = None

        wandb.init(
            project=config.wandb_project_name,
            notes=self.c.get("general", "wandb_notes", fallback=None),
            id=self.run_name,
            resume=resume,
            allow_val_change=True
        )
        logging.getLogger("wandb").setLevel(logging.DEBUG)
        # wandb.save(CONFIG_FILE)

    def configure_wandb(self):
        if config.wandb_project_name is not None:
            self.wandb_log = self.c.getboolean(
                "general", "wandb_enable", fallback=False)
        else:
            self.wandb_log = False

        if self.wandb_log:
            self.initialize_wandb()

    def set_device(self):
        # Set the device to use for training
        if self.cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.pin_memory = True
        else:
            self.device = torch.device("cpu")
            self.pin_memory = False
        self.n_gpus = torch.cuda.device_count()

    def display_device_info(self):
        if self.n_gpus > 1:
            logger.info(
                f"Using device '{self.device}' with {self.n_gpus} GPUs")
        else:
            logger.info(f"Using {self.device}")

    def print_params(self):
        for attribute, value in vars(self).items():
            logger.info(f"{attribute:>24s}: {value}")

            # Load parameters to wandb
            if self.wandb_log:
                if self.load_epoch == 0:
                    wandb.config[attribute] = value
                else:
                    wandb.config.update(
                        {attribute: value}, allow_val_change=True)

            # TODO: AGGIUNGERE TUTTI I PARAMS NECESSARI DA PRINTARE
