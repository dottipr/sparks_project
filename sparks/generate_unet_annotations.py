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
19.10.2021  Copiato .py file in una nuova cartella per processare le nuove
            annotazioni per il training usando gli smoothed video di Miguel
            invece dei video originali
07.02.2022  Generato annotazioni per video corretti [13,22,34-35] e video
            aggiunti al training [30,32]
23.02.2022  Corretto bug (video importato con valori interi) e perfezionato
            nonmaxima_suppression. Procedimento di nuovo utilizzando i video
            originali.

'''

import os
import glob
import argparse
import imageio
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import ellipsoid



def nonmaxima_suppression(img, min_dist_xy, min_dist_t,
                          return_mask=False, threshold=0.5):

    smooth_img = ndi.gaussian_filter(img, 2)
    #smooth_img = img

    min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
    #dilated = ndi.grey_dilation(smooth_img,
    dilated = ndi.maximum_filter(smooth_img,
                                footprint=min_dist)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima

def final_mask(mask, radius1=2.5, radius2=3.5, ignore_ind=2): # SLOW
    dt = ndi.distance_transform_edt(1 - mask)
    new_mask = np.zeros(mask.shape, dtype=np.int64)
    new_mask[dt < radius2] = ignore_ind
    new_mask[dt < radius1] = 1

    return new_mask

def get_new_mask(video, mask, min_dist_xy, min_dist_t,
                 radius_event=3, radius_ignore=2, ignore_index=4,
                 return_loc=False, return_loc_mask=False):

    # get spark centres
    if 1 in mask:
        sparks_mask = np.where(mask == 1, video, 0).astype(np.float32)
        sparks_loc, sparks_mask = nonmaxima_suppression(sparks_mask,
                                                        min_dist_xy, min_dist_t,
                                                        return_mask=True)

        print("\t\tNum of sparks:", len(sparks_loc))

        if return_loc:
            return sparks_loc

        sparks_mask = final_mask(sparks_mask, radius1=radius_event,
                             radius2=radius_event+radius_ignore,
                             ignore_ind=ignore_index)
    else:
        if return_loc:
            return 0

        return mask

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

    old_mask_folder = "original_masks"
    #video_folder = "smoothed_movies"
    video_folder = "original_movies"
    out_folder = "unet_masks"

    # events paramenters
    radius_event = 3
    radius_ignore = 1
    #radius_ignore = 3
    ignore_index = 4

    # physiological params
    pixel_size = 0.2 # 1 pixel = 0.2 um x 0.2 um
    min_dist_xy = round(1.8 / pixel_size) # min distance in space between sparks
    time_frame = 6.8 # 1 frame = 6.8 ms
    min_dist_t = round(20 / time_frame) # min distance in time between sparks


    for id in args.sample_ids:
        print("Processing mask "+id+"...")

        old_mask_name = id+"_mask.tif"
        old_mask_path = os.path.join(old_mask_folder, old_mask_name)
        old_mask = np.asarray(imageio.volread(old_mask_path)).astype('int')

        #video_name = id+"_smoothed_video.tif"
        video_name = id+".tif"
        video_path = os.path.join(video_folder, video_name)
        video = np.asarray(imageio.volread(video_path))

        if old_mask.shape != video.shape:
            #video_name = id+"_cut_smoothed_video.tif"
            video_name = id+"_cut.tif"
            video_path = os.path.join(video_folder, video_name)
            video = np.asarray(imageio.volread(video_path))

        print("\tOld values:", np.unique(old_mask))

        mask = get_new_mask(video=video, mask=old_mask,
                            min_dist_xy=min_dist_xy, min_dist_t=min_dist_t,
                            radius_event=radius_event,
                            radius_ignore=radius_ignore)

        print("\tNew values:", np.unique(mask))

        out_path = os.path.join(out_folder, id+"_video_mask.tif")
        imageio.volwrite(out_path, np.uint8(mask))
