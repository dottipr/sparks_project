# Detection and Classification of Local Ca²⁺ Release Events in Cardiomyocytes Using 3D-UNet Neural Network

## Parameters summary

### Fixed parameters


These parameters are consistent throughout the whole project and are not printed.

🠊 They are hardcoded in the dictionary `params{}` in the script _training.py_.


1. General parameters
	* verbosity: always using level 3
	* logfile: not using it because using wandb (configure it in the future, when project is finished and uploaded in github)
	* wandb_project_name: project name used in WandB
	* output_relative_path: relative path where the model parameters are saved (currently saving everything in the "runs/" directory)
2. Dataset parameters
	* ignore_index: index ignored by loss function, it is 4 by definition in the dataset
	* num_classes: it is 4 by definition of the problem (sparks, puffs, waves, BG)
	* ndims: it is 3, since using 3-dimensional data


### Printed parameters


These parameters are important to be recovered after the training for comparison with the successive trainings.

🠊 They are loaded in the dictionary `params{}` in the script _training.py_ from the configuration file.

1. Training parameters
	* run_name: name of the run, it will be used to save the testing results, to synchronise with wandb (and tensorboard), to save the model parameters and load them to continue the training/analyse the model (if not using a `load_run_name`)
	* load_run_name: print and load this only if it is used, use this if wanting to continue the training of a saved model with a different configuration
	* load_epoch: if different from 0, continue the training of a saved model from the parameters saved at this epoch
	* train_epochs: number of iterations that will be trained
	* criterion: select criterion for training, possible options are 'nll_loss', 'focal_loss', 'lovasz_softmax' and 'sum_losses'
	* lr_start: initial learning rate
	* ignore_frames_loss: number of frames at the beginning at end of the UNet input that are ignored by the loss function
	* gamma: print and load this only if using 'focal_loss' or 'sum_losses'
	* w: print and load this only if using 'sum_losses'
	* cuda: if true, train UNet on GPU, if available
2. Dataset parameters
	* relative_path: relative directory containing the dataset samples and labels
	* dataset_size: use 'full' to run the script using the whole dataset, use 'minimal' to run the script using only one sample for training and one for testing (for debugging purposes)
	* batch_size: batch size used by the dataloader
	* num_workers: number of workers used in the dataloader
	* data_duration: number of frames of the data movie sampled as input for the UNet
	* data_step: step between two consecutive input for the UNet sampled from a data movie
	* data_smoothing: smoothing that is applied to the dataset samples as a preprocessing procedure, possible values are '2d', '3d' or 'none'
	* norm_video: normalisation that is applied to the dataset samples as a preprocessing step, possible values are 'abs_max', 'chunk', 'movie' or 'none'
	* remove_background: remove background from dataset samples as a preprocessing procedure, possible values are 'moving', 'average' or 'none'
	* only_sparks: if true, remove puffs and waves from labels, print only if true
	* noise_data_augmentation: if true, apply some denoising/add noise randomly to the dataset movies as data augmentation
	* sparks_type: preprocessing applied to the spark labels, options are 'raw' or 'peaks'
3. NN architecture parameters

	* nn_architecture: used to select which network architecture to use, for the moment, the options are 'pablos_unet' and 'github_unet'
	* unet_steps: depth of the UNet architecture
	* first_layer_channels: number of output channels in the first convolutional layer
	* temporal_reduction: if true, process the input into a simple convolutional NN to reduce its temporal dimension, while increasing the number of channels of the input given to the UNet
	* num_channels: usually 1, it has a bigger value if e.g. using temporal reduction
	* dilation: if true, compute dilated convolution in the UNet model, not printed if not used
	* border_mode: possible values are 'same' and 'valid', if 'same' pad sampled input so that also border points are used during the convolution
	* batch_normalization: type of normalisation applied at the end of each block, for the moment possible values are 'batch' or 'none'
	* initialize_weights: if true, initialise network weights using a normal distribution
	* attention: ...


### Other parameters

These parameters are not important to be printed for comparison with the successive trainings.

🠊 They are loaded in the script _training.py_ from the configuration file directly where they are used

1. General parameters
	* wandb_enable: if true, log the training outputs and testing metrics on WandB
	* training: true to train the network
	* testing: true to run testing function at the end of training
2. Training parameters
	* save_every: number of iterations between model saving
	* test_every: number of iterations between testing
	* print_every: number of iterations between logging current training loss


### Currently not used

These parameters are currently unused, they are only present in the *config_....ini* files.

* fixed_threshold: detection threshold for test function, currently not used
* t_detection_sparks: threshold applied to sparks detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* t_detection_puffs: threshold applied to puffs detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* t_detection_waves: threshold applied to waves detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* sparks_min_radius: minimal radius below which spark predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
* puffs_min_radius: minimal radius below which puff predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
* waves_min_radius: minimal radius below which wave predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
