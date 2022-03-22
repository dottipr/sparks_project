"""
Functions needed to preprocess the csv files, create the structure of data in
the spark datasets and compute the weights of the network.
"""

import os
import imageio
import glob

import numpy as np
import torch
from torch import nn
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from PIL import Image

import torch

from metrics_tools import nonmaxima_suppression


__all__ = ["get_chunks",
           "random_flip",
           "compute_class_weights",
           "weights_init",
           "get_times",
           "get_fps",
           "video_spline_interpolation",
           "remove_avg_background",
           "shrink_mask",
           "get_new_voxel_label",
           "final_mask",
           "get_new_mask",
           "load_movies",
           "load_movies_ids",
           "load_annotations",
           "load_predictions",
           "load_predictions_all_trainings"
           ]


### functions for unet masks creation ###


def final_mask(mask, radius1=2.5, radius2=3.5, ignore_ind=2): # SLOW
    '''
    add annotation region around spark peaks
    '''
    dt = ndi.distance_transform_edt(1 - mask)
    new_mask = np.zeros(mask.shape, dtype=np.int64)
    new_mask[dt < radius2] = ignore_ind
    new_mask[dt < radius1] = 1

    return new_mask


def get_new_mask(video, mask, min_dist_xy, min_dist_t,
                 radius_event=3, radius_ignore=2, ignore_index=4,
                 sigma=2, return_loc=False, return_loc_mask=False):
    '''
    from raw segmentation masks get masks where sparks are annotated by peaks
    '''

    # get spark centres
    if 1 in mask:
        sparks_mask = np.where(mask == 1, video, 0).astype(np.float32)
        sparks_loc, sparks_mask = nonmaxima_suppression(sparks_mask,
                                                        min_dist_xy, min_dist_t,
                                                        return_mask=True,
                                                        sigma=sigma)

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


### functions for data preproccesing ###


def get_chunks(video_length, step, duration):
    n_blocks = ((video_length-duration)//(step))+1

    return np.arange(duration)[None,:] + step*np.arange(n_blocks)[:,None]


def random_flip(x, y):

    if np.random.uniform() > 0.5:
        x = x[..., ::-1]
        y = y[..., ::-1]

    if np.random.uniform() > 0.5:
        x = x[..., ::-1, :]
        y = y[..., ::-1, :]

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    return x, y


def remove_avg_background(video):
    # remove average background
    avg = np.mean(video, axis = 0)
    return np.add(video, -avg)


### functions related to U-Net hyperparameters ###


def compute_class_weights(dataset, w0=1, w1=1, w2=1, w3=1):
    # For 4 classes
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with torch.no_grad():
        for _,y in dataset:
            count0 += np.count_nonzero(y==0)
            count1 += np.count_nonzero(y==1)
            count2 += np.count_nonzero(y==2)
            count3 += np.count_nonzero(y==3)

    total = count0 + count1 + count2 + count3

    w0_new = w0*total/(4*count0) if count0 != 0 else 0
    w1_new = w1*total/(4*count1) if count1 != 0 else 0
    w2_new = w2*total/(4*count2) if count2 != 0 else 0
    w3_new = w3*total/(4*count3) if count3 != 0 else 0

    weights = np.array([w0_new, w1_new, w2_new, w3_new])

    return np.float64(weights)


def weights_init(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        stdv = np.sqrt(2/m.weight.size(1))
        m.weight.data.normal_(m.weight, std=stdv)


### functions for video resampling ###


def get_times(video_path):
    # get times at which video frames where sampled
    description = Image.open(video_path).tag[270][0].split('\r\n')
    description  = [line.split('\t') for line in description]
    description = [[int(i) if i.isdigit() else i for i in line] for line in description]
    description = [d for d in  description if isinstance(d[0], int)]
    return np.array([float(line[1]) for line in description])


def get_fps(video_path):
    # compute estimated video fps value wrt sampling times deltas
    times = get_times(video_path)
    deltas = np.diff(times)
    return 1/np.mean(deltas)


def video_spline_interpolation(video, video_path, new_fps=150):
    # interpolate video wrt new sampling times
    frames_time = get_times(video_path)
    f = interp1d(frames_time, video, kind='linear', axis=0)
    assert(len(frames_time) == video.shape[0])
    frames_new = np.linspace(frames_time[0], frames_time[-1], int(frames_time[-1]*new_fps))
    return f(frames_new)


### functions for temporal reduction ###


def shrink_mask(mask, num_channels):
    # input is an annotation mask with the number of channels of the unet
    # output is a shrinked mask where :
    # {0} -> 0
    # {0, i}, {i} -> i, i = 1,2,3
    # {0, 1, i}, {1, i} -> 1, i = 2,3
    # {0, 2 ,3}, {2, 3} -> 2
    # and each voxel in the output corresponds to 'num_channels' voxels in
    # the input

    assert mask.shape[0] % num_channels == 0, \
    "in shrink_mask the duration of the mask is not a multiple of num_channels"

    # get subtensor of duration 'num_channels'
    sub_masks = np.split(mask, mask.shape[0]//num_channels)

    #print(sub_masks[0].shape)
    #print(len(sub_masks))

    new_mask = []
    # for each subtensor get a single frame
    for sub_mask in sub_masks:
        new_frame = np.array([[get_new_voxel_label(sub_mask[:,y,x]) for x in range(sub_mask.shape[2])] for y in range(sub_mask.shape[1])])
        #print(new_frame.shape)
        new_mask.append(new_frame)

    new_mask = np.stack(new_mask)
    return new_mask

def get_new_voxel_label(voxel_seq):
    # voxel_seq is a vector of 'num_channels' elements
    # {0} -> 0
    # {0, i}, {i} -> i, i = 1,2,3
    # {0, 1, i}, {1, i} -> 1, i = 2,3
    # {0, 2 ,3}, {2, 3} -> 3
    #print(voxel_seq)

    if np.max(voxel_seq == 0):
        return 0
    elif 1 in voxel_seq:
        return 1
    elif 3 in voxel_seq:
        return 3
    else:
        return np.max(voxel_seq)


################################ Loading utils #################################

'''
Use this functions to load predictions (ys, sparks, puffs, preds) or just
annotations ys
'''

def load_movies(data_folder):
    '''
    Load all movies in data_folder whose name start with [0-9].

    data_folder: folder where movies are saved, movies are saved as
                 "[0-9][0-9]*.tif"
    '''
    xs_all_trainings = {}

    xs_filenames = sorted(glob.glob(os.path.join(data_folder,
                                                 "[0-9][0-9]*.tif")))

    for f in xs_filenames:
        video_id = os.path.split(f)[1][:2]
        xs_all_trainings[video_id] = np.asarray(imageio.volread(f))

    return xs_all_trainings

def load_movies_ids(data_folder, ids):
    '''
    Same as load_movies but load only movies corresponding to a given list of
    indices.

    data_folder: folder where movies are saved, movies are saved as
                 "[0-9][0-9]*.tif"
    ids : list of movies IDs (of the form "[0-9][0-9]")
    '''
    xs_all_trainings = {}

    xs_filenames = [os.path.join(data_folder,movie_name)
                    for movie_name in os.listdir(data_folder)
                    if movie_name.startswith(tuple(ids))]

    for f in xs_filenames:
        video_id = os.path.split(f)[1][:2]
        xs_all_trainings[video_id] = np.asarray(imageio.volread(f))

    return xs_all_trainings

def load_annotations(data_folder, mask_names="video_mask"):
    '''
    open and process annotations (original version, sparks not processed)

    data_folder: folder where annotations are saved, annotations are saved as
                 "[0-9][0-9]_video_mask.tif"
    '''
    ys_all_trainings = {}

    ys_filenames = sorted(glob.glob(os.path.join(data_folder,
                                            "[0-9][0-9]_"+mask_names+".tif")))

    for f in ys_filenames:
        video_id = os.path.split(f)[1][:2]
        ys_all_trainings[video_id] = np.asarray(imageio.volread(f)).astype('int')

    return ys_all_trainings

def load_predictions(training_name, epoch, metrics_folder):
    '''
    open and process annotations (where sparks have been processed), predicted
    sparks, puffs and waves for a given training name
    !!! the predictions movies have to be saved in metrics_folder for the given
        training name !!!

    training_name: saved training name
    epoch: training epoch whose predictions have to be loaded
    metrics_folder: folder where predictions and annotations are saved,
                    annotations are saved as "[0-9]*_ys.tif"
                    sparks are saved as "<base name>_[0-9][0-9]_sparks.tif"
                    puffs are saved as "<base name>_[0-9][0-9]_puffs.tif"
                    waves are saved as "<base name>_[0-9][0-9]_waves.tif"
    '''

    # Import .tif files as numpy array
    base_name = os.path.join(metrics_folder,training_name+"_"+str(epoch)+"_")

    if "temporal_reduction" in training_name:
        # need to use annotations from another training
        # TODO: implement a solution ....
        print('''!!! method is using temporal reduction, processed annotations
                     have a different shape !!!''')


    # get predictions and annotations filenames
    ys_filenames = sorted(glob.glob(base_name+"[0-9][0-9]_video_ys.tif"))
    sparks_filenames = sorted(glob.glob(base_name+"[0-9][0-9]_video_sparks.tif"))
    puffs_filenames = sorted(glob.glob(base_name+"[0-9][0-9]_video_puffs.tif"))
    waves_filenames = sorted(glob.glob(base_name+"[0-9][0-9]_video_waves.tif"))

    # create dictionaires to store loaded data for each movie
    training_ys = {}
    training_sparks = {}
    training_puffs = {}
    training_waves = {}

    for y,s,p,w in zip(ys_filenames,
                       sparks_filenames,
                       puffs_filenames,
                       waves_filenames):

        # get movie name
        video_id = y.replace(base_name,"")[:2]

        ys_loaded = np.asarray(imageio.volread(y)).astype('int')
        training_ys[video_id] = ys_loaded

        if "temporal_reduction" in training_name:
            # repeat each frame 4 times
            print("training using temporal reduction, extending predictions...")
            s_preds = np.asarray(imageio.volread(s))
            p_preds = np.asarray(imageio.volread(p))
            w_preds = np.asarray(imageio.volread(w))

            # repeat predicted frames x4
            s_preds = np.repeat(s_preds,4,0)
            p_preds = np.repeat(p_preds,4,0)
            w_preds = np.repeat(w_preds,4,0)

            # TODO: can't crop until annotations loading is fixed
            # if original length %4 != 0, crop preds
            #if ys_loaded.shape != s_preds.shape:
            #    duration = ys_loaded.shape[0]
            #    s_preds = s_preds[:duration]
            #    p_preds = p_preds[:duration]
            #    w_preds = w_preds[:duration]

            # TODO: can't check until annotations loading is fixed
            #assert ys_loaded.shape == s_preds.shape
            #assert ys_loaded.shape == p_preds.shape
            #assert ys_loaded.shape == w_preds.shape

            training_sparks[video_id] = s_preds
            training_puffs[video_id] = p_preds
            training_waves[video_id] = w_preds
        else:
            training_sparks[video_id] = np.asarray(imageio.volread(s))
            training_puffs[video_id] = np.asarray(imageio.volread(p))
            training_waves[video_id] = np.asarray(imageio.volread(w))

    return training_ys, training_sparks, training_puffs, training_waves

def load_predictions_all_trainings(training_names, epochs, metrics_folder):
    '''
    open and process annotations (where sparks have been processed), predicted
    sparks, puffs and waves for a list of training names
    !!! the predictions movies have to be saved in metrics_folder for the given
        training name !!!

    training_names: list of saved training names
    epochs: list of training epochs whose predictions have to be loaded
            corresponding to the training names
    metrics_folder: folder where predictions and annotations are saved,
                    annotations are saved as "[0-9][0-9]_ys.tif"
                    sparks are saved as "<base name>_[0-9][0-9]_sparks.tif"
                    puffs are saved as "<base name>_[0-9][0-9]_puffs.tif"
                    waves are saved as "<base name>_[0-9][0-9]_waves.tif"
    '''
    # dicts with "shapes":
    # num trainings (dict) x num videos (dict) x video shape
    ys = {}
    s = {} # sparks
    p = {} # puffs
    w = {} # waves

    for name, epoch in zip(training_names, epochs):
        ys[name],s[name],p[name],w[name] = load_predictions(training_name,
                                                            epoch,
                                                            metrics_folder)

    return ys, s, p, w
