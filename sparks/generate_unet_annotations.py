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

REMARKS:
01.03.2022  Questo codice ora è adattato a PC232.

'''

import os
import argparse
import imageio
import numpy as np

from dataset_tools import get_new_mask, load_movies_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate u-net segmentations")

    parser.add_argument("sample_ids", metavar='sample IDs', nargs='+',
            help="select sample ids for which .tif segmentation will be generated")

    args = parser.parse_args()

    print(args.sample_ids)

    raw_data_directory = os.path.join("..", "data", "raw_data_and_processing")
    old_mask_folder = os.path.join(raw_data_directory,"original_masks")
    #video_folder = "smoothed_movies"
    video_folder = os.path.join(raw_data_directory,"original_movies")
    out_folder = os.path.join(raw_data_directory,"unet_masks")

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

        video = load_movies_ids(video_folder, [id])[id]
        print("\tVideo shape", video.shape)

        print("\tOld values:", np.unique(old_mask))

        mask = get_new_mask(video=video, mask=old_mask,
                            min_dist_xy=min_dist_xy, min_dist_t=min_dist_t,
                            radius_event=radius_event,
                            radius_ignore=radius_ignore)

        print("\tNew values:", np.unique(mask))

        out_path = os.path.join(out_folder, id+"_video_mask.tif")
        imageio.volwrite(out_path, np.uint8(mask))
