'''
26.09.2022

Script contenente tutte le funzioni necessarie per processare
le preds della UNet.

E.g. raw preds --> binary preds --> separated events

Le funzioni sono specifiche per le preds della UNet, quelle
che vanno bene anche per i samples del dataset (preproccessing)
sono invece nello script "dataset_tools.py".

Contiene anche funzioni per visualizzare le preds con Napari
(e.g. creazione della colormap).
'''

import numpy as np
import glob
import os
import imageio
import napari
import matplotlib.pyplot as plt
from matplotlib import cm
import vispy.color
import math
from PIL import Image

import cc3d
import skimage
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from metrics_tools import simple_nonmaxima_suppression
from dataset_tools import detect_spark_peaks



############################# Visualisation tools ##############################


# define function to obtain discrete Colormap instance that can be used by Napari

def get_discrete_cmap(name='viridis', lut=16):
    # create original cmap
    segmented_cmap = cm.get_cmap(name=name, lut=16)

    # get colors
    colors = segmented_cmap(np.arange(0,segmented_cmap.N))

    # get new discrete cmap
    cmap = vispy.color.Colormap(colors, interpolation='zero')

    return cmap

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


def get_spark_signal(video,
                     sparks_labelled,
                     center,
                     radius,
                     context_duration,
                     return_info = False):
    # video:             is the original video sample
    # sparks_labelled:   is a mask containing the segmentation of the
    #                    spark events (1 integer for every event)
    # center:            [t y x] is the center of the selected event to plot
    # radius:            is the radius of the considered region around the
    #                    center the spark for averaging
    # context_duration:  is the number of frames included in the analysis before
    #                    and after the event

    t,y,x = center
    event_idx = sparks_labelled[t,y,x] # event_idx = 1,2,...,n_sparks

    assert event_idx != 0, (
    "given center does not correspond to any event in the given labelled mask")

    loc = ndi.measurements.find_objects(sparks_labelled)[event_idx-1]

    assert loc[0].start <= t and loc[0].stop > t, "something weird is wrong"

    # get mask representing sparks location (with radius and context)
    start = loc[0].start - context_duration
    stop = loc[0].stop + context_duration

    start = max(0,start)
    stop = min(video.shape[0],stop)
    signal_mask = create_signal_mask(*sparks_labelled.shape,
                                     start, stop,
                                     (x,y), radius)


    frames = np.arange(start,stop)
    signal = np.average(video[start:stop],
                        axis=(1,2),
                        weights=signal_mask[start:stop])

    if return_info:
        return frames, signal, (y,x), loc[0].start, loc[0].stop

    return frames, signal

def get_spark_2d_signal(video, slices, coords, spatial_context, sigma = 2, return_info = False):
    # video : original video
    # slices : slices in the 3 dimensions of a given spark (ROI)
    # coords [t y x]: center of a given spark
    # spatial_context : extend ROI corresponding to spark
    # sigma : for gaussian filtering

    # TODO: add assertion to check that coords are inside slices

    t,y,x = coords
    t_slice, y_slice, x_slice = slices

    y_start = max(0, y_slice.start-spatial_context)
    y_end = min(video.shape[1], y_slice.stop+spatial_context)

    x_start = max(0, x_slice.start-spatial_context)
    x_end = min(video.shape[2], x_slice.stop+spatial_context)

    signal_2d = video[t, y_start:y_end, x_start:x_end]

    # average over 3 frames
    #signal_2d_avg = video_array[t-1:t+1, y_start:y_end, x_start:x_end]
    #signal_2d_avg = np.average(signal_2d_avg, axis=0)

    # smooth signal
    #signal_2d_gaussian = ndimage.gaussian_filter(signal_2d, 2) # Best

    if return_info:
        y_frames = np.arange(y_start, y_end)
        x_frames = np.arange(x_start, x_end)

        return t, y, x, y_frames, x_frames, signal_2d

    return signal_2d #signal_2d_gaussian


################ Visualisation tools (for colored segmentation) ################


def paste_segmentation_on_video(video, colored_mask):
    # video is a RGB video, list of PIL images
    # colored_mask is a RGBA video, list of PIL images
    for frame,ann in zip(video, colored_mask):
        frame.paste(ann, mask = ann.split()[3])

def add_colored_segmentation_to_video(segmentation,video,color,transparency=50):
    # segmentation is a binary array
    # video is a RGB video, list of PIL images
    # color is a list of 3 RGB elements

    # convert segmentation into a colored mask
    #mask_shape = (*(segmentation.shape), 4)
    #colored_mask = np.zeros(mask_shape, dtype=np.uint8)
    r,g,b = color
    colored_mask = np.stack((r*segmentation, g*segmentation, b*segmentation, transparency*segmentation), axis=-1)
    colored_mask = colored_mask.astype(np.uint8)

    colored_mask = [Image.fromarray(frame).convert('RGBA') for frame in colored_mask]

    paste_segmentation_on_video(video, colored_mask)
    return video

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





############################ UNet processing tools #############################


def get_argmax_segmented_output(preds, get_classes=True):
    '''
    preds are the (exponential) raw outputs of the unet for each class:
    [background, sparks, waves, puffs] (4 x duration x 64 x 512)
    '''

    argmax_classes = np.argmax(preds, axis=0)

    if not get_classes:
        return argmax_classes

    preds = {}
    preds['sparks'] = np.where(argmax_classes==1,1,0)
    preds['waves'] = np.where(argmax_classes==2,1,0)
    preds['puffs'] = np.where(argmax_classes==3,1,0)

    #imageio.volwrite("TEST_argmax.tif", np.uint8(argmax_classes))

    return preds, argmax_classes


def get_argmax_segmentation_otsu(preds, get_classes=True, debug=False):
    '''
    preds are the (exponential) raw outputs of the unet for each class:
    dict with keys 'sparks', 'puffs', 'waves', 'background'

    compute otsu threshold with respect to the sum of positive predictions
    (i.e., sparks+puffs+waves) and remove preds below that threshold,
    then get argmax predictions on thresholded UNet output
    '''

    # compute threshold on summed predicted events
    sum_preds = 1 - preds['background'] # everything but the background
    t_otsu = threshold_otsu(sum_preds)
    if debug:
        print("\tEvents detection threshold:",t_otsu)

    # get binary mask of valid predictions
    binary_sum_preds = sum_preds > t_otsu

    # mask out removed events from UNet preds
    masked_class_preds = binary_sum_preds * ([preds['background'],
                                             preds['sparks'],
                                             preds['waves'],
                                             preds['puffs']])

    # get argmax of classes
    # if get_classes==False, return an array with values in {0,1,2,3}
    # if get_classes==True, return a pair (argmax_preds, classes_preds) where
    # argmax_preds is a dict with keys 'sparks', 'waves' and 'puffs' and
    # classes_preds is an array with values in {0,1,2,3}

    return get_argmax_segmented_output(preds=masked_class_preds,
                                       get_classes=get_classes)


def get_separated_events(argmax_preds, movie, sigma,
                         connectivity, connectivity_mask,
                         return_sparks_loc = False,
                         debug = False):
    '''
    movie:              input movie
    sigma:              sigma valued used for nonmaxima suppression and
                        watershed separation of sparks
    argmax_preds:       segmented UNet output (dict with keys 'sparks', 'puffs',
                        'waves')
    connectivity:       int, define how puffs and waves are separated
    connectivity_mask:  3d matrix, define how sparks are separated

    Given the segmented output, separate each class into event instances.

    Return a dict with keys 'sparks', 'puffs' and 'waves' where each entry is an
    array with labelled events (from 1 to n_events).

    Using watershed separation algorithm to separate spark events.
    '''

    # separate CCs in puff and wave classes
    ccs_class_preds = {class_name: cc3d.connected_components(class_argmax_preds,
                                                             connectivity=connectivity,
                                                             return_N=False
                                                             )
                      for class_name, class_argmax_preds in argmax_preds.items()
                      if class_name != 'sparks'}

    # compute spark peaks locations
    loc, mask_loc = simple_nonmaxima_suppression(img=movie,
                                                 maxima_mask=argmax_preds['sparks'],
                                                 min_dist=connectivity_mask,
                                                 return_mask=True,
                                                 threshold=0.,
                                                 sigma=sigma)

    if debug:
        print(f"\tNumber of sparks detected by nonmaxima suppression: {len(loc)}")

    # compute smooth version of input video
    smooth_xs = ndi.gaussian_filter(movie, sigma=sigma)

    # compute watershed separation
    markers, _ = ndi.label(mask_loc)

    split_event_mask = watershed(image=-smooth_xs,
                                 markers=markers,
                                 mask=argmax_preds['sparks'],
                                 connectivity=3,
                                 compactness=1
                                )

    # check if all connected components have been labelled
    all_ccs_labelled = np.all(split_event_mask.astype(bool)
                              == argmax_preds['sparks'].astype(bool))

    if not all_ccs_labelled:
        if debug:
            print("\tNot all sparks were labelled, computing missing events...")
            print("\tNumber of sparks before correction:", np.max(split_event_mask))

        # get number of labelled events
        n_split_events = np.max(split_event_mask)

        # if not all CCs have been labelled, obtain unlabelled CCs and split them
        missing_sparks = np.logical_xor(split_event_mask.astype(bool),
                                        argmax_preds['sparks'].astype(bool))

        # separate unlabelled CCs and label them
        labelled_missing_sparks = cc3d.connected_components(missing_sparks,
                                                            connectivity=connectivity,
                                                            return_N=False)

        # increase labels by number of sparks already present
        labelled_missing_sparks = np.where(labelled_missing_sparks,
                                           labelled_missing_sparks+n_split_events,
                                           0)

        # merge sparks with peaks and sparks without them
        split_event_mask += labelled_missing_sparks

        # get peak location of missing sparks and add it to peaks lists
        missing_sparks_ids = list(np.unique(labelled_missing_sparks))
        missing_sparks_ids.remove(0)
        for spark_id in missing_sparks_ids:
            spark_roi_xs =  np.where(labelled_missing_sparks==spark_id,
                                     smooth_xs, 0)

            peak_loc = np.unravel_index(spark_roi_xs.argmax(),
                                        spark_roi_xs.shape)

            loc.append(list(peak_loc))

        # assert that now all CCs have been labelled
        all_ccs_labelled = np.all(split_event_mask.astype(bool)
                                  == argmax_preds['sparks'].astype(bool))

        if debug:
            print("\tNumber of sparks after correction:", np.max(split_event_mask))

    assert all_ccs_labelled, "Some sparks CCs haven't been labelled!"

    # check that event IDs are ordered and consecutive
    assert len(np.unique(split_event_mask))-1 == np.max(split_event_mask), \
           f"spark IDs are not consecutive: {np.unique(split_event_mask)}"
    assert len(np.unique(ccs_class_preds['puffs']))-1 == np.max(ccs_class_preds['puffs']), \
           f"puff IDs are not consecutive: {np.unique(ccs_class_preds['puffs'])}"
    assert len(np.unique(ccs_class_preds['waves']))-1 == np.max(ccs_class_preds['waves']), \
           f"wave IDs are not consecutive: {np.unique(ccs_class_preds['waves'])}"

    separated_events = {'sparks': split_event_mask,
                        'puffs': ccs_class_preds['puffs'],
                        'waves': ccs_class_preds['waves']
                       }

    if return_sparks_loc:
        return separated_events, loc

    return separated_events
