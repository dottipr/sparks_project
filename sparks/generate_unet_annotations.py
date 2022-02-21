'''
21.01.2021

Usare questo script per creare delle masks .tif che possono essere usate
durante il training della u-net come annotazioni dei samples.

Input:  masks .tif contenenti le ROIs degli eventi con valori da 1 a 4

Output: masks .tif dove puffs, waves e zone da ignorare rimangono invariati, ma
        gli sparks sono indicati dal loro centro

UPDATES:
18.01.2021  Generati nuove annotations dove la ignore_region per gli sparks è
            molto più grande (1 --> 3)
31.08.2021  Generate annotazioni con ignore_region == 1 per i nuovi video
            [12-14; 18-20; 29-30; 38-46]

TODO:
15.02.2022  Tenere sempre aggiornato rispetto al file salvato localmente (su
            switchdrive), ad esempio ora il file usa gli smoothed videos invece
            del video originale per calcolare il centro degli sparks.
'''

import os
import glob

import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import imageio
from scipy import ndimage, spatial

import matplotlib.pyplot as plt

import argparse

def nonmaxima_suppression(img, return_mask=False, neighborhood_radius=5, threshold=0.5):

    smooth_img = ndi.gaussian_filter(img, 2)
    dilated = ndi.grey_dilation(smooth_img, (neighborhood_radius,) * img.ndim)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima

def final_mask(mask, radius1=2.5, radius2=3.5, ignore_ind=2): # SLOW
    dt = ndimage.distance_transform_edt(1 - mask)
    new_mask = np.zeros(mask.shape, dtype=np.int64)
    new_mask[dt < radius2] = ignore_ind
    new_mask[dt < radius1] = 1

    return new_mask

def get_new_mask(video, mask, radius_event=3, radius_ignore=2, ignore_index=4,
                 return_loc=False, return_loc_mask=False):

    # get spark centres
    sparks_mask = np.where(mask == 1, video, 0).astype(np.float32)
    sparks_loc, sparks_mask = nonmaxima_suppression(sparks_mask, return_mask=True)

    if return_loc:
        return sparks_loc

    sparks_mask = final_mask(sparks_mask, radius1=radius_event,
                         radius2=radius_event+radius_ignore,
                         ignore_ind=ignore_index)

    # remove sparks from old mask
    no_sparks_mask = np.where(mask == 1, 0, mask)

    # create new mask
    new_mask = np.where(sparks_mask != 0, sparks_mask, no_sparks_mask)

    if return_loc_mask:
        return sparks_loc, new_mask

    return new_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate u-net segmentations")

    parser.add_argument("sample_ids", metavar='sample IDs', nargs='+',
            help="select sample ids for which .tif segmentation will be generated")

    args = parser.parse_args()

    print(args.sample_ids)

    tif_folder = "tif_files"
    out_folder = "unet_masks"

    # events paramenters
    radius_event = 3
    radius_ignore = 1
    #radius_ignore = 3
    ignore_index = 4
    
    for id in args.sample_ids:
        old_mask_name = id+"_corrected_mask.tif"
        old_mask_path = os.path.join(tif_folder, old_mask_name)
        old_mask = np.asarray(imageio.volread(old_mask_path)).astype('int')

        video_name = id+"_video.tif"
        video_path = os.path.join(tif_folder, video_name)
        video = np.asarray(imageio.volread(video_path)).astype('int')

        if old_mask.shape != video.shape:
            video_name = id+"_cut_video.tif"
            video_path = os.path.join(tif_folder, video_name)
            video = np.asarray(imageio.volread(video_path)).astype('int')

        print("Processing mask "+id+"...")
        print("\tOld values:", np.unique(old_mask))

        mask = get_new_mask(video, old_mask,
                            radius_event=radius_event,
                            radius_ignore=radius_ignore)

        print("\tNew values:", np.unique(mask))

        out_path = os.path.join(out_folder, id+"_unet_mask.tif")
        imageio.volwrite(out_path, np.uint8(mask))
