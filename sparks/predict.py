#!/usr/bin/env python3

import sys
import os
import logging
import logging.handlers
import argparse
import pathlib
import configparser

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import imageio

import unet
from dataset_tools import compute_class_weights_puffs, weights_init
from datasets import MaskTEMPTestDataset


BASEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASEDIR, "config.ini")

logger = logging.getLogger(__name__)
# disable most wandb logging, since we don't care about it here
logging.getLogger('wandb').setLevel(logging.ERROR)


# we don't care about gradients, and retaining them explodes the memory usage
@torch.no_grad()
def predict(network: nn.Module, dataset: Dataset, quick: bool = False) -> torch.Tensor:
    # Since the dataset acts more like a generator than a list, we can't yield the first item
    # without breaking access to other elements. Thus, the container must be initialized later
    predictions = None

    # retain some contextual information (in the form of previous and following frames)
    # by slicing results according to a frame overlap
    overlap = (dataset.duration - dataset.step) // 2
    assert overlap % 2 == 0, f"Frame overlap must be an even number; is {overlap}"

    # TODO: have this use a dataloader
    # dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    for i, (chunk, labels) in enumerate(dataset):
        if args.quick and 0 < i < (len(dataset) - 1):
            # skipping all but the first and last frame
            continue

        logger.debug(f"Sample {i+1:>3}/{len(dataset):>3}")
        if predictions is None:
            # initialize mask container with the size from the first chunk
            predictions = torch.zeros(
                (
                    1 + 1 + 4,                                      # input + labels + number of classes
                    len(dataset) * dataset.step + 2 * overlap,      # time, i.e. every video frame
                    *(chunk.shape[1:])                              # the actual frames, height x width
                ), dtype=torch.float32, device=device)
            logger.debug(f"Created prediction container of shape {predictions.shape}")

        # for every frame, cut the predictions into the range [-overlap, overlap], except for
        # - the first item (start at the first frame)
        # - the last item (end at the last frame)
        # additionally, keep track of the absolute locations of the data in the input video
        if i == 0:
            logger.debug(f"First sample, adding extra {overlap} frames")
            slice_start = 0
            slice_end = -overlap
            abs_start = 0
            abs_end = dataset.step + overlap
        else:
            slice_start = overlap
            slice_end = -overlap
            abs_start = i * dataset.step + overlap
            abs_end = abs_start + dataset.step
            if i == len(dataset) - 1:
                logger.debug(f"Last sample, adding extra {overlap} frames")
                slice_end = None  # slice to and including the end
                abs_end += overlap

        logger.debug(
            f"Slicing results: [{slice_start}:{slice_end}] "
            f"mapping to time frames [{abs_start:>5}:{abs_end:<5}]")

        chunk_t = torch.Tensor(chunk).to(device)
        labels_t = torch.Tensor(labels).to(device)

        prediction = network(chunk_t[None, None])

        # remove empty first dim
        # note that predictions still have one more dim than data & labels!
        prediction_squeeze = torch.squeeze(prediction, dim=0)
        data_squeeze = torch.squeeze(chunk_t, dim=0)
        label_squeeze = torch.squeeze(labels_t, dim=0)

        # this line is slightly different from ...[:, start:end] - I couldn't find a way to ask for a slice
        # which would have actually included the last element in the standard [start:end] notation.
        # instead, slice(start, end) does what I need by accepting None
        prediction_slice = prediction_squeeze[:, slice(slice_start, slice_end)]
        data_slice = data_squeeze[slice(slice_start, slice_end)]
        label_slice = label_squeeze[slice(slice_start, slice_end)]

        time_length = prediction_slice.shape[1]

        logger.debug(
            f"Mangling prediction {prediction.shape} into sliced {prediction_slice.shape}, "
            f"spanning {time_length} time units")

        # poor man's unittest to assert correct slicing of video chunks
        assert time_length == (abs_end - abs_start)
        if i == 0 or i == len(dataset) - 1:
            # first and last entries are longer
            assert time_length == dataset.step + overlap
        else:
            assert time_length == dataset.step

        predictions[0, abs_start:abs_end, :, :] = data_slice
        predictions[1, abs_start:abs_end, :, :] = label_slice
        predictions[2:, abs_start:abs_end, :, :] = prediction_slice

    return predictions


def parse_predictions(predictions: torch.Tensor, desc: str, output: str) -> None:
    logger.info(f"Generating output files in {os.path.abspath(output)}")

    # produce a singleton output, containing all outputs stacked, i.e. stacked along the vertical pixel axis
    # since .cat() requires a collection of tensors, we create that using torch's views along the first dim;
    # and finally, we have to remove the empty (one-dimensional) first dimension by squeeze()
    predictions_stacked = torch.cat(torch.split(predictions, 1, dim=0), dim=2).squeeze()
    fn = os.path.join(output, f"{desc}_stacked.tif")
    # TODO: doing this twice is wasteful; instead, exp().numpy() once and use it for every subsequent volwrite
    logger.debug(f"Prediction has shape {predictions.shape}, stack along output axis has shape {predictions_stacked.shape}")
    data_np = torch.exp(predictions_stacked).detach().cpu().numpy()
    imageio.volwrite(fn, data_np)

    # produce an output file per output
    for class_label, data in zip(
        ("input", "labels", "unknown", "sparks", "waves", "puffs"),
        predictions
    ):
        fn = os.path.join(output, f"{desc}_{class_label}.tif")
        logger.debug(f"Building {fn} from data in shape {data.shape}")
        data_np = torch.exp(data).detach().cpu().numpy()
        imageio.volwrite(fn, data_np)


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
        '-s', '--state',
        type=str,
        default=c.get("state", "file", fallback=None),
        help="Load pretrained model from this file"
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=c.getint("general", "batch_size", fallback="1"),
        help="Use this batch size for training & evaluation."
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=c.get("general", "output", fallback="output"),
        help="Move output files to this directory"
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=145.0,
        help=""
    )

    parser.add_argument(
        'input',
        type=str,
        nargs=1,
        help="Input video file, used as input for the model to produce predictions"
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

    parser.add_argument(
        '-q', '--quick',
        action="store_true",
        default=False,
        help="Run on as little data as possible while still touching as much functionality as possible for debugging."
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
        file_handler = logging.handlers.RotatingFileHandler(
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

    # detect CUDA devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    assert len(args.input) == 1, f"More than one input provided: {len(args.input)}"
    input_file = pathlib.Path(args.input[0])
    name_for_ds, ext = os.path.splitext(input_file.name)

    dataset = MaskTEMPTestDataset(
        # TODO: Fix path building in dataset
        base_path=input_file.parent.parent,
        video_name=name_for_ds,
        smoothing='2d',
        step=c.getint("data", "step"),
        duration=c.getint("data", "chunks_duration"),
        remove_background=c.getboolean("data", "remove_background")
    )

    logger.info(f"Loaded dataset with {len(dataset)} samples")

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

    # configure unet
    unet_config = unet.UNetConfig(
        steps=c.getint("data", "step"),
        num_classes=c.getint("network", "num_classes"),
        ndims=c.getint("network", "ndims"),
        border_mode=c.get("network", "border_mode"),
        batch_normalization=c.getboolean("network", "batch_normalization")
    )
    unet_config.feature_map_shapes((c.getint("data", "chunks_duration"), 64, 512))
    network = nn.DataParallel(unet.UNetClassifier(unet_config)).to(device)

    if c.getboolean("network", "initialize_weights", fallback="no"):
        network.apply(weights_init)

    # load model state
    if not os.path.isfile(args.state):
        raise RuntimeError(f"State is not a file: {args.state}")

    logger.info(f"Loading state from {args.state}")
    network.load_state_dict(torch.load(args.state, map_location=device))

    # feed data to the network
    network.eval()
    predictions = predict(network, dataset, args.quick)
    parse_predictions(predictions, desc=name_for_ds, output=args.output)
