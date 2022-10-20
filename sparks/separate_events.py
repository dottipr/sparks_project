'''
13.09.2022

Run this script to separate the predicted events, starting from the UNet raw
predictions.

It will save the results in the directory
"trainings_validation/<training name>/instance_segmentation"

For each movie, it will save:
XX_pred_class_label.tif (movie with predicted class label for each pixel)
XX_pred_event_label.tif (movie with enumerated separated events)
'''

import numpy as np
import math
import glob
import os
import imageio
import napari
import matplotlib.pyplot as plt

import cc3d
import skimage
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from dataset_tools import load_annotations_ids
from metrics_tools import get_argmax_segmented_output

BASEDIR = os.path.abspath('')
BASEDIR


####################### Functions and global parameters ########################


PIXEL_SIZE = 0.2 # 1 pixel = 0.2 um x 0.2 um
global MIN_DIST_XY
MIN_DIST_XY = round(1.8 / PIXEL_SIZE) # min distance in space between sparks
TIME_FRAME = 6.8 # 1 frame = 6.8 ms
global MIN_DIST_T
MIN_DIST_T = round(20 / TIME_FRAME) # min distance in time between sparks


def simple_nonmaxima_suppression(img,maxima_mask=None,
                                 min_dist=None,
                                 return_mask=False, threshold=0.5, sigma=2):
    '''
    Extract local maxima from input array (t,x,y).
    img :           input array
    maxima_mask :   if not None, look for local maxima only inside the mask
    min_dist :      define minimal distance between two peaks
    return_mask :   if True return both masks with maxima and locations, if
                    False only returns locations
    threshold :     minimal value of maximum points
    sigma :         sigma parameter of gaussian filter
    '''
    img = img.astype(np.float64)

    # handle min_dist connectivity mask
    if min_dist is None:
        min_dist = 1

    if np.isscalar(min_dist):
        c_min_dist = ndi.generate_binary_structure(img.ndim, min_dist)
    else:
        c_min_dist = np.array(min_dist, bool)
        if c_min_dist.ndim != img.ndim:
            raise ValueError("Connectivity dimension must be same as image")

    if maxima_mask is not None:
        # mask out region from img with mask
        masked_img = np.where(maxima_mask, img, 0.)

        # smooth masked input image
        smooth_img = ndi.gaussian_filter(masked_img, sigma=sigma)

    else:
        smooth_img = ndi.gaussian_filter(img, sigma=sigma)

    # search for local maxima
    dilated = ndi.maximum_filter(smooth_img,
                                 footprint=c_min_dist)

    if maxima_mask is not None:
        # hyp: maxima belong to maxima mask
        masked_smooth_img = np.where(maxima_mask, smooth_img, 0.)
        argmaxima = np.logical_and(smooth_img == dilated,
                                   masked_smooth_img > threshold)
    else:
        argmaxima = np.logical_and(smooth_img == dilated,
                                   smooth_img > threshold)


    argwhere = np.argwhere(argmaxima)
    argwhere = np.array(argwhere, dtype=float)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


################################## Load files ##################################
'''
Load predictions from files generated when running the test function during
training.

They are saved in `runs\<training_name>\predictions` and the filename template
is `<training_name>_<movie_id>_<preds class/xs/ys/...>.tif`.

The epoch is not explicitly mentionned, but it depends on the last epoch of the
last model that has been run.
'''

# assuming these are the movie IDs whose predictions are available...
movie_ids = ["05","10","15","20","25","32","34","40","45"]

# Select training name to load #
training_name = 'TEMP_new_annotated_peaks_physio'

# Configure input and output directories #
data_dir = os.path.join("runs", training_name, "predictions")
out_dir = os.path.join()"trainings_validation", training_name, "instance_segmentation")

# Load movies #
xs_filenames = {movie_id: os.path.join(data_dir,
                                       training_name+"_"+movie_id+"_xs.tif")
                                       for movie_id in movie_ids}
xs = {movie_id: np.asarray(imageio.volread(f))
      for movie_id, f in xs_filenames.items()}

# Load annotations #
# annotations using during training (from predictions folder)
ys_filenames = {movie_id: os.path.join(data_dir,
                                       training_name+"_"+movie_id+"_ys.tif")
                                       for movie_id in movie_ids}
ys = {movie_id: np.asarray(imageio.volread(f)).astype('int')
      for movie_id, f in ys_filenames.items()}

# Load annotations #
# predictions created by selected model
sparks_filenames = {movie_id: os.path.join(data_dir,
                                           training_name+"_"+movie_id+"_sparks.tif")
                                           for movie_id in movie_ids}
puffs_filenames = {movie_id: os.path.join(data_dir,
                                          training_name+"_"+movie_id+"_puffs.tif")
                                          for movie_id in movie_ids}
waves_filenames = {movie_id: os.path.join(data_dir,
                                          training_name+"_"+movie_id+"_waves.tif")
                                          for movie_id in movie_ids}

sparks = {movie_id: np.asarray(imageio.volread(f))
          for movie_id, f in sparks_filenames.items()}
puffs = {movie_id: np.asarray(imageio.volread(f))
         for movie_id, f in puffs_filenames.items()}
waves = {movie_id: np.asarray(imageio.volread(f))
         for movie_id, f in waves_filenames.items()}

# Load annotated event instances #
# from dataset directory
dataset_dir =  os.path.join("..", "data", "sparks_dataset")
annotated_events = load_annotations_ids(data_folder=dataset_dir,
                                        ids=movie_ids,
                                        mask_names="event_label"
                                        )


########################### Process each test video ############################

for sample_id in movie_ids:
    sample = {'xs' : xs[sample_id],
              'ys' : ys[sample_id]}

    preds = {'sparks' : sparks[sample_id],
             'puffs' : puffs[sample_id],
             'waves' : waves[sample_id],
             'background' : 1-sparks[sample_id]-puffs[sample_id]-waves[sample_id]}

    # Extract events from summed predictions #
    sum_preds = 1 - preds['background']
    t_otsu = threshold_otsu(sum_preds) # detection threshold
    binary_sum_preds = sum_preds > t_otsu

    # Get argmax classes in binary predictions #
    # select argmax class inside binary prediction mask

    # get mask containing network output for each class in binary preds
    masked_class_preds = binary_sum_preds * [preds['background'],
                                             preds['sparks'],
                                             preds['waves'],
                                             preds['puffs']]

    # get argmax of classes inside binary preds
    # argmax_preds is a dict with keys 'sparks', 'waves' and 'puffs'
    argmax_preds, classes_preds = get_argmax_segmented_output(preds=masked_class_preds,
                                                              get_classes=True)

    # Get connected components for each class #
    # separate CCs in each class
    conn = 26
    # much faster than ndi.label:
    ccs_class_preds = {class_name: cc3d.connected_components(class_argmax_preds,
                                                             connectivity=conn,
                                                             return_N=False
                                                            )
                      for class_name, class_argmax_preds in argmax_preds.items()}


    ### Watershed separation of spark preds ###
    '''
    Only separate spark preds.
    Using directly the argmax prediction of the sparks class.
    APPLY WATERSHED SEPARATION USING SMOOTH MOVIE AS INPUT AND NONMAXIMA
    SUPPRESSION PEAKS
    '''

    # define mask for minimal allowed distance between peaks
    radius = math.ceil(MIN_DIST_XY/2)
    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
    disk = x**2+y**2 <= radius**2
    connectivity_mask = np.stack([disk]*(MIN_DIST_T), axis=0)

    # detect spark peaks locations in smooth movie
    sigma = 3
    loc, mask_loc = simple_nonmaxima_suppression(img=sample['xs'],
                                                 maxima_mask=argmax_preds['sparks'],
                                                 min_dist=connectivity_mask,
                                                 return_mask=True,
                                                 threshold=0.,
                                                 sigma=sigma)

    # compute smooth version of input video
    smooth_xs = ndi.gaussian_filter(sample['xs'], sigma=sigma)

    # compute watershed separation
    markers, _ = ndi.label(mask_loc)

    split_event_mask = watershed(image=-smooth_xs,
                                 markers=markers,
                                 mask=argmax_preds['sparks'],
                                 connectivity=3,
                                 compactness=1
                                )

    # check if all connected components have been labelled
    all_ccs_labelled = np.all(split_event_mask.astype(bool) == argmax_preds['sparks'].astype(bool))

    if not all_ccs_labelled:
        # get number of labelled events
        n_split_events = np.max(split_event_mask)

        # if not all CCs have been labelled, obtain unlabelled CCs and split them
        missing_sparks = np.logical_xor(split_event_mask.astype(bool),
                                        argmax_preds['sparks'].astype(bool))

        # separate unlabelled CCs and label them
        connectivity=26
        labelled_missing_sparks = cc3d.connected_components(missing_sparks,
                                                            connectivity=connectivity,
                                                            return_N=False)

        # increase labels by number of sparks already present
        labelled_missing_sparks = np.where(labelled_missing_sparks,
                                           labelled_missing_sparks+n_split_events,
                                           0)

        # merge sparks with peaks and sparks without them
        split_event_mask += labelled_missing_sparks

        # assert that now all CCs have been labelled
        all_ccs_labelled = np.all(split_event_mask.astype(bool) == argmax_preds['sparks'].astype(bool))

    assert all_ccs_labelled, "Some sparks CCs haven't been labelled!"

    '''
    ################# Compute spark peaks in predicted events ##################
    # TODO: USE THIS IN METRICS COMPUTATION, IF NECESSARY
    from dataset_tools import detect_spark_peaks

    coords_pred = detect_spark_peaks(movie=sample['xs'],
                   event_mask=split_event_mask,
                   class_mask=split_event_mask.astype(bool),
                   sigma=2,
                   max_filter_size=10,
                   return_mask=False)'''

    # Create dict containing separated events for each class #
    separated_events = {'sparks': split_event_mask,
                        'puffs': ccs_class_preds['puffs'],
                        'waves': ccs_class_preds['waves']
                       }


    ###################### Save predicted events on disk #######################
    # TODO: ...
