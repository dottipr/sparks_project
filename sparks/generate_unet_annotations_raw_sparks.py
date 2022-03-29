'''
29.03.2021

Usare questo script per creare delle masks .tif che possono essere usate
durante il training della u-net come annotazioni dei samples.

Input:  masks .tif contenenti le ROIs degli eventi con valori da 1 a 4

Output: masks .tif dove ogni evento presenta una ignore_region attorno a sé

UPDATES:

REMARKS:

'''

import os
import argparse
import imageio
import numpy as np

from dataset_tools import get_new_mask_raw_sparks, load_movies_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate u-net segmentations")

    parser.add_argument("sample_ids", metavar='sample IDs', nargs='+',
            help="select sample ids for which .tif segmentation will be generated")

    args = parser.parse_args()

    print(args.sample_ids)

    raw_data_directory = os.path.join("..", "data", "raw_data_and_processing")
    old_mask_folder = os.path.join(raw_data_directory,"original_masks")
    out_folder = os.path.join(raw_data_directory,"unet_masks_raw_sparks")
    os.makedirs(out_folder, exist_ok=True)

    # events paramenters
    radius_ignore_sparks = 1
    radius_ignore_puffs = 2
    radius_ignore_waves = 3
    ignore_index = 4


    for id in args.sample_ids:
        print("Processing mask "+id+"...")

        old_mask_name = id+"_mask.tif"
        old_mask_path = os.path.join(old_mask_folder, old_mask_name)
        old_mask = np.asarray(imageio.volread(old_mask_path)).astype('int')

        print("\tOld values:", np.unique(old_mask))

        mask = get_new_mask_raw_sparks(mask=old_mask,
                            radius_ignore_sparks=radius_ignore_sparks,
                            radius_ignore_puffs=radius_ignore_puffs,
                            radius_ignore_waves=radius_ignore_waves,
                            ignore_index=ignore_index
                            )

        print("\tNew values:", np.unique(mask))

        out_path = os.path.join(out_folder, id+"_video_mask_raw_sparks.tif")
        imageio.volwrite(out_path, np.uint8(mask))
