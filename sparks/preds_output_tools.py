'''
Script con funzioni che permettono di processare i video delle preds da salvare
sul disco.
'''

import os
import datetime
import imageio
import itertools

import numpy as np
import math
from PIL import Image
from scipy import spatial, optimize

from metrics_tools import correspondences_precision_recall, get_sparks_locations_from_mask, process_spark_prediction
from dataset_tools import load_movies_ids, load_predictions, load_annotations



############### TOOLS FOR LABELLING COLORED SPARK PEAKS ON MOVIE ###############


def paste_annotations_on_video(video, colored_mask):
    # video is a RGB video, list of PIL images
    # colored_mask is a RGBA video, list of PIL images
    for frame,ann in zip(video, colored_mask):
        frame.paste(ann, mask = ann.split()[3])


def add_colored_annotations_to_video(annotations,video,color,transparency=50,radius=4):
    # annotations is a list of t,x,y coordinates
    # video is a RGB video, list of PIL images
    # color is a list of 3 RGB elements
    mask_shape = (len(video), video[0].size[1], video[0].size[0], 4)
    colored_mask = np.zeros(mask_shape, dtype=np.uint8)
    for pt in annotations:
        colored_mask = color_ball(colored_mask,pt,radius,color,transparency)
    colored_mask = [Image.fromarray(frame).convert('RGBA') for frame in colored_mask]

    paste_annotations_on_video(video, colored_mask)
    return video


def l2_dist(p1,p2):
    # p1 = (t1,y1,x1)
    # p2 = (t2,y2,x2)
    t1,y1,x1 = p1
    t2,y2,x2 = p2
    return math.sqrt(math.pow((t1-t2),2)+math.pow((y1-y2),2)+math.pow((x1-x2),2))


def ball(c,r):
    # r scalar
    # c = (t,y,x)
    # returns coordinates c' around c st dist(c,c') <= r
    t,y,x = c
    t_vect = np.linspace(t-r,t+r, 2*r+1, dtype = int)
    y_vect = np.linspace(y-r,y+r, 2*r+1, dtype = int)
    x_vect = np.linspace(x-r,x+r, 2*r+1, dtype = int)

    cube_idxs = list(itertools.product(t_vect,y_vect,x_vect))
    ball_idxs = [pt for pt in cube_idxs if l2_dist(c, pt) <= r]

    return ball_idxs


def color_ball(mask,c,r,color,transparency=50):
    color_idx = ball(c,r)
    # mask boundaries
    duration, height, width, _ = np.shape(mask)

    for t,y,x in color_idx:
        if 0 <= t and t < duration and 0 <= y and y < height and 0 <= x and x < width:
            mask[t,y,x] = [*color, transparency]

    return mask


####################### TOOLS FOR WRITING VIDEOS ON DISK #######################


def write_videos_on_disk(training_name, video_name, path="predictions",
                         xs=None, ys=None, preds=None):
    '''
     Write all videos on disk
     xs : input video used by network
     ys: segmentation video used in loss function
     preds : all u-net preds [bg preds, sparks preds, puffs preds, waves preds]
    '''
    out_name_root = training_name + "_" + video_name + "_"

    if not isinstance(xs, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "xs.tif"),
                                      xs)
    if not isinstance(ys, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "ys.tif"),
                                      np.uint8(ys))
    if not isinstance(preds, type(None)):
        imageio.volwrite(os.path.join(path, out_name_root + "sparks.tif"),
                                      np.exp(preds[1]))
        imageio.volwrite(os.path.join(path, out_name_root + "waves.tif"),
                                      np.exp(preds[2]))
        imageio.volwrite(os.path.join(path, out_name_root + "puffs.tif"),
                                      np.exp(preds[3]))


def write_colored_sparks_on_disk(training_name, video_name,
                                 paired_real, paired_pred,
                                 false_positives, false_negatives,
                                 path="predictions", xs=None,
                                 movie_shape=None):
    '''
     Write input video with colored paired sparks and text file with sparks
     coordinates on disk.

     training_name, video_name : used to save output on disk
     paired_real : list of coordinates [t,y,x] of paired annotated sparks
     paired_pred : list of coordinates [t,y,x] of paired predicted sparks
     false_positives : list of coordinates [t,y,x] of wrongly predicted sparks
     false_negatives : list of coordinates [t,y,x] of not found annotated sparks
     path : directory where output will be saved
     xs: input video used by network, if None, save sparks on white background
     movie_shape : if input movie xs is None, provide video shape (t,y,x)
    '''
    out_name_root = training_name + "_" + video_name + "_"

    if not isinstance(xs, type(None)):
        sample_video = 255*(xs/xs.max())
    else:
        assert not isinstance(movie_shape, type(None)), "Provide movie shape if not providing input movie."
        sample_video = 255*np.ones(movie_shape)

    # compute colored sparks mask
    transparency = 45
    rgb_video = [Image.fromarray(frame).convert('RGB') for frame in sample_video]

    annotated_video = add_colored_annotations_to_video(paired_real, rgb_video, [0,255,0], 0.8*transparency)
    annotated_video = add_colored_annotations_to_video(paired_preds, annotated_video, [0,255,200], 0.8*transparency)
    annotated_video = add_colored_annotations_to_video(false_positives, annotated_video, [255,255,0], transparency)
    annotated_video = add_colored_annotations_to_video(false_negatives, annotated_video, [255,0,0], transparency)

    annotated_video = [np.array(frame) for frame in annotated_video]

    # set saved movies filenames
    white_background_fn = "_white_backgroud" if white_background else ""
    out_name_root = training_name + "_" + video_name + white_background_fn + "_"

    # save video on disk
    imageio.volwrite(os.path.join(path, f"{out_name_root}colored_sparks.tif"),
                     annotated_video)

    # write sparks locations to file
    file_path = os.path.join(path, f"{out_name_root}_sparks_location.txt")

    with open(file_path, 'w') as f:
        f.write(f"{datetime.datetime.now()}\n\n")
        f.write(f"Paired annotations and preds:\n")
        for p_true, p_preds in zip(paired_real, paired_preds):
            f.write(f"{list(map(int, p_true))} {list(map(int, p_preds))}\n")
        f.write(f"\n")
        f.write(f"Unpaired preds (false positives):\n")
        for f_p in false_positives:
            f.write(f"{list(map(int, f_p))}\n")
        f.write(f"\n")
        f.write(f"Unpaired annotations (false negatives):\n")
        for f_n in false_negatives:
            f.write(f"{list(map(int, f_n))}\n")
