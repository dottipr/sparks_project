'''
25.11.2021

Get sparks locations from given .tif unet spark class prediction
Write sparks locations in .csv file
'''

import sys
import os
import logging
import logging.handlers
import argparse
import pathlib
import configparser
from typing import List

import numpy as np
import imageio
import csv

from metrics_tools import process_spark_prediction, empty_marginal_frames


BASEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASEDIR, "config.ini")

logger = logging.getLogger(__name__)
# disable most wandb logging, since we don't care about it here
logging.getLogger('wandb').setLevel(logging.ERROR)


def create_csv(filename, positions):
    #N = positions.shape[0]
    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['frame','x','y'])
        #for n in range(N):
        #    filewriter.writerow([positions[n,0], positions[n,1], positions[n,2]])
        for loc in positions:
            filewriter.writerow([loc[0], loc[2], loc[1]])
            logger.info(f"Location {[loc[0], loc[2], loc[1]]} written to .csv file")


if __name__ == "__main__":

    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")

    parser = argparse.ArgumentParser("Get sparks locations from predictions.")

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
        '-o', '--output',
        type=str,
        default=c.get("general", "output", fallback="output"),
        help="Move output files to this directory"
    )

    parser.add_argument(
        'input',
        type=str,
        nargs=1,
        help="Input video file, used as input for the model to produce predictions"
    )


    '''
    # TODO: if ground truth is provided, print precision and recall (& other
    #       metrics)
    parser.add_argument(
        '-gt', '--gt_available',
        action="store_true",
        default=c.get("predict", "gt_available", fallback=False),
        help="Compare predictions with given gt_available"
    )
    '''

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


    # extract input file name to better describe the output.
    input_desc, _ = os.path.splitext(os.path.basename(args.input[0]))

    # open predictions and remove frames at boundaries
    pred_mask = np.asarray(imageio.volread(pathlib.Path(args.input[0])))
    pred_mask = empty_marginal_frames(pred_mask,
                                      c.getint("data","ignore_frames_loss"))

    # process predictions
    sparks_list = process_spark_prediction(pred_mask,
                                           c.getfloat("processing",
                                                      "t_detection_sparks",
                                                      fallback=0.9),
                                           c.getint("processing",
                                                    "neighborhood_radius_sparks",
                                                    fallback=5),
                                           c.getint("processing",
                                                    "min_radius_sparks",
                                                    fallback=3))

    logger.info(f"Writing sparks locations to .csv file in {os.path.abspath(args.output)}")

    csv_filename = os.path.join(args.output, f"{input_desc}_locations.csv")
    create_csv(csv_filename, sparks_list)
