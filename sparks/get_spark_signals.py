# 30.11.2020
# Functions to extract the signal of the sparks present in the annotations
# Run this script to plot a few samples

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy import ndimage


def create_circular_mask(h, w, center, radius):
    # h : image height
    # w : image width
    # center : center of the circular mask (x_c, y_c) !!
    # radius : radius of the circular mask
    # returns a circular mask of given radius around given center

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius

    return mask


def create_signal_mask(t, h, w, start, stop, center, radius):
    # t : video duration
    # h : video height
    # w : video width
    # start : first frame
    # stop : last frame (not included!)
    # center : center of the circular mask (x_c,y_c)
    # radius : radius of the circular mask

    start = max(0,start)
    stop = min(t,stop)

    frame_mask = create_circular_mask(h,w,center,radius)

    video_mask = np.zeros((t,h,w), dtype=bool)
    video_mask[start:stop] = frame_mask

    return video_mask


def get_spark_signal(video, sparks_labelled, idx, radius, context_duration,
                     return_info = False):
    # video :            the original video sample
    # sparks_labelled :  mask containing the segmentation of the spark events
    #                    (1 integer for every event)
    # idx :              index of the selected event to plot
    #                    (0,1,....,num_sparks-1)
    # radius :           radius of the considered region around the center the
    #                    spark for averaging
    # context_duration : number of frames included in the analysis before and
    #                    after the event
    # returns signal average and list of corresponding frames


    assert (idx < sparks_labelled.max()),(
    f"given idx is too large, video contains only {sparks_labelled.max()} events")

    loc = ndimage.measurements.find_objects(sparks_labelled)[idx]
    roi = sparks_array[loc].astype(bool)

    # extract center (x_c,y_c) of the event
    center = ndimage.measurements.center_of_mass(np.sum(roi, axis=0))
    center = tuple(map(lambda x: isinstance(x, float) and round(x), center))#int
    y,x = int(loc[1].start+center[0]), int(loc[2].start+center[1])

    # get mask representing sparks location (with radius and context)
    start = max(0, loc[0].start - context_duration)
    stop = min(video.shape[0], loc[0].stop + context_duration)
    signal_mask = create_signal_mask(*sparks_array.shape, start, stop,
                                     (x,y), radius)

    frames = np.arange(start,stop)
    signal = np.average(video_array[start:stop], axis=(1,2),
                        weights=signal_mask[start:stop])

    if return_info:
        return frames, signal, (y,x), loc[0].start, loc[0].stop

    return frames, signal


if __name__ == "__main__":

    # Import .tif events file as numpy array
    data_path = os.path.join("..","data","annotation_masks")
    sample_name = "130918_C_ET-1.tif"
    events_name = os.path.join(data_path,"masks_test",sample_name)
    video_name = os.path.join(data_path,"videos_test",sample_name)

    events_array = np.asarray(imageio.volread(events_name)).astype('int')
    video_array = np.asarray(imageio.volread(video_name)).astype('int')


    # only analyse spark events, get rid of the rest
    sparks_array = np.where(events_array==1,1,0)


    # separate events: assign an integer to every connected component
    sparks_labelled, n_sparks = ndimage.measurements.label(sparks_array,
                                                           np.ones((3,3,3),
                                                           dtype=np.int))

    # get a list with subarrays indices of sparks_labelled for every event
    find_sparks = ndimage.measurements.find_objects(sparks_labelled)

    # define radius around event center and context duration
    radius = 3
    context_duration = 30

    # Plot few signal samples
    plt.rcParams.update({'font.size': 6})

    plt.figure(figsize=(20,10))
    plt.suptitle("Signal around some sample sparks", fontsize=10)

    for idx in range(10):
        frames, signal, (y,x), start, stop = get_spark_signal(video_array,
                                                              sparks_labelled,
                                                              idx,radius,
                                                              context_duration,
                                                              return_info=True)

        ax = plt.subplot(2,5,idx+1)
        ax.set_title(f"Spark at position ({x},{y}) at frames {start}-{stop}")
        ax.axvspan(start, stop, facecolor='green', alpha=0.3)
        plt.plot(frames,signal,color='darkgreen')


    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.show()
