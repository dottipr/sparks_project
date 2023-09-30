# Detection and Classification of Local Ca²⁺ Release Events in Cardiomyocytes Using 3D-UNet Neural Network

## Project's directories organization
TODO: Idea presa da chatGPT.. adattare il mio codice e poi adattare questa lista
project_root/
│
├── data/
│   ├── raw/
│   │   ├── dataset1/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── dataset2/
│   │       ├── train/
│   │       └── test/
│   ├── processed/
│   │   ├── dataset1/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── dataset2/
│   │       ├── train/
│   │       └── test/
│   ├── annotations/
│   │   ├── dataset1/
│   │   ├── dataset2/
│   └── ...
│
├── code/
│   ├── notebooks/
│   ├── src/
│   │   ├── data_preprocessing/
│   │   ├── model/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── utils/
│   ├── config/
│   ├── experiments/
│   └── ...
│
├── experiments/
│   ├── experiment1/
│   ├── experiment2/
│   └── ...
│
├── logs/
│   ├── training_logs/
│   └── ...
│
├── models/
│   ├── saved_models/
│   └── ...
│
├── results/
│   ├── metrics/
│   ├── visualizations/
│   └── ...
│
├── environment.yml
├── README.md
├── requirements.txt
└── .gitignore

## How to clone this repo

Use command:
```
git clone --recurse-submodules -j8 git@github.com:priscacheese/sparks_project.git
```

(See: [How do I "git clone" a repo, including its submodules?](https://stackoverflow.com/questions/3796927/how-do-i-git-clone-a-repo-including-its-submodules))


## REMARKS:
Some parts of the code are optimized to accept different datasets with different classes etc, but some others aren't.  
For example, some visualization tools only work for our specific dataset.
## List of main Python scripts (status at 18/10/2022)

Short explanation of the purposes of each script present in this project.

1. *architecture.py*: in questo file scrivo tutte le variazioni della UNet architecture oppure altre architectures completamente diverse
2. *dataset_tools.py*: script con tutte le funzioni che sono specifiche per la creazione dei datasets
	* 🠊 magari il file sarebbe da "merge" con *dataset.py* (?)
	* controllare se le funzioni sono piuttosto da includere nella class `SparkDataset`
	* muovere le funzioni più generiche in altri script
3. *datasets.py*: file dove ho implementato il dataset [movie(s)-> chunks]
4. *generate_unet_annotations_raw_sparks.py*: qui processo le raw annotations per ottenere un input per la UNet
4. *generate_unet_annotations.py*: stessa cosa, ma produco degli input per la network dove gli sparks sono particolarmente processati (e.g. peaks)
	* 🠊 probabilmente da "merge" con il file precedente, aggiungendo un parametro per scegliere che tipo di processing voglio fare
	* idea: ottenere un file unico per fare il processing "raw annotations" -> "movies ready for UNET", cioè tutte quei processing che non voglio ripetere ogni volta che genero un dataset per il training (tipo aggiungere la ignore region, visto che ci va una vita)
5. *get_sparks_from_preds.py*: file che apre un output della network e salva le locations degli sparks in un file .csv
	* 🠊 probabilmente inutile ora, tasferire le funzioni utili in uno script con tutte le funzioni per scrivere cose sul disco e eliminare questo script
6. *load_trainings_predict.py*: load un trained model della UNet e produce delle raw merged predictions che vengono poi salvate sul disco
	* 🠊 da tenere e controllare solo se e come aggiornarlo (e.g., per fare diversi tipi di assemblaggi dai chunks)
7. *metrics_tools.py*: funzioni che vengono usate per calcolare le metrics (cioè funzioni che prendono come input le raw preds/segmented preds/events instances)
8. *new_unet.py*: questa è un implementazione della UNet che ho trovato su github, ha dei parametri in più che si possono cambiare rispetto a quella di Pablo, ma non sono sicura che abbia davvero dei vantaggi, tengo il file solo per sicurezza e se caso lo elimino alla fine del progetto
9. ~~*other_losses.py*~~ 🠊 *custom_losses.py*: in questo file salvo tutte le loss functions che provo e che non sono già implementate in Pytorch
	* 🠊 cambiare nome del file e adattare scripts dove viene importato
10. ~~*preds_output_tools.py*~~ 🠊 "disc_output_tools.py": script con funzioni per salvare vari outputs sul disco (video, immagini, csv files...)
	* 🠊 cambiare nome del file e adattare scripts dove viene importato
11. *preds_processing_tools.py*: script che contiene tutte le funzioni necessarie per processare gli output della UNet (partendo dalle raw preds dei chunks, a come li rimetto assieme, fino all'ottenimento di segmentation masks e instances detection)
	* 🠊 pensare bene se le funzioni che sono contenute qui starebbero meglio in un altro script
	* magari a un certo punto si potrà chiamare *inference_tools.py*, ma per intanto lascio così
	* !!! magari bisogna mantenere qui le funzioni per l'inferenza e avere un file separato con le funzioni che vanno bene sia per il pre- che per il post-processing !!! (ad es., *data_processing.py*)
12. *save_metrics_to_json.py*: in questo file apro input movies, annotations e predictions per calcolare delle metrics
	* 🠊 molto probabilmente outdated, vedere se ci sono delle funzioni da tenere e salvare da un'altra parte e poi da eliminare
13. *save_results_to_json.py*: in questo file apro input movies, annotations e predictions per calcolare tp, fp, fn da salvare sul disco per poi calcolare le metrics
	* 🠊 molto probabilmente outdated, vedere se ci sono delle funzioni da tenere e salvare da un'altra parte e poi da eliminare
14. *separate_events.py*: qui stavo cominciando a riassumere il notebook di Jupyter corrispondente, però non sono sicura che abbia senso avere un file che non sia su Jupyter...
15. *track_objects_from_preds.py*: file scritto per separare gli eventi
	* 🠊 molto probabilmente outdated, vedere se ci sono delle funzioni da tenere e salvare da un'altra parte e poi da eliminare
16. *training_tools.py*: script che contiene la training e testing functions
17. *training.py*: script da eseguire per fare il training della network

## Parameters summary

### Fixed parameters


These parameters are consistent throughout the whole project and are not printed.

🠊 They are hardcoded in the dictionary `params{}` in the script _training.py_.


1. General parameters
	* **verbosity**: always using level 3
	* **logfile**: not using it because using wandb (configure it in the future, when project is finished and uploaded in github)
	* **wandb_project_name**: project name used in WandB
	* **output_relative_path**: relative path where the model parameters are saved (currently saving everything in the "runs/" directory)
2. Dataset parameters
	* **ignore_index**: index ignored by loss function, it is 4 by definition in the dataset
	* **num_classes**: it is 4 by definition of the problem (sparks, puffs, waves, BG)
	* **ndims**: it is 3, since using 3-dimensional data


### Printed parameters


These parameters are important to be recovered after the training for comparison with the successive trainings.

🠊 They are loaded in the dictionary `params{}` in the script _training.py_ from the configuration file.

1. Training parameters
	* **run_name**: name of the run, it will be used to save the testing results, to synchronise with wandb (and tensorboard), to save the model parameters and load them to continue the training/analyse the model (if not using a `load_run_name`)
	* **load_run_name**: print and load this only if it is used, use this if wanting to continue the training of a saved model with a different configuration
	* **load_epoch**: if different from 0, continue the training of a saved model from the parameters saved at this epoch
	* **train_epochs**: number of iterations that will be trained
	* **criterion: select criterion for training, possible options are 'nll_loss', 'focal_loss', 'lovasz_softmax' and 'sum_losses'
	* **lr_start**: initial learning rate
	* **ignore_frames_loss**: number of frames at the beginning at end of the UNet input that are ignored by the loss function
	* **gamma**: print and load this only if using 'focal_loss' or 'sum_losses'
	* **w**: print and load this only if using 'sum_losses'
	* **cuda**: if true, train UNet on GPU, if available
2. Dataset parameters
	* **relative_path**: relative directory containing the dataset samples and labels
	* **dataset_size**: use 'full' to run the script using the whole dataset, use 'minimal' to run the script using only one sample for training and one for testing (for debugging purposes)
	* **batch_size**: batch size used by the dataloader
	* **num_workers**: number of workers used in the dataloader
	* **data_duration**: number of frames of the data movie sampled as input for the UNet
	* **data_step**: step between two consecutive input for the UNet sampled from a data movie
	* **data_smoothing**: smoothing that is applied to the dataset samples as a preprocessing procedure, possible values are '2d', '3d' or 'none'
	* **norm_video**: normalisation that is applied to the dataset samples as a preprocessing step, possible values are 'abs_max', 'chunk', 'movie' or 'none'
	* **remove_background**: remove background from dataset samples as a preprocessing procedure, possible values are 'moving', 'average' or 'none'
	* **only_sparks**: if true, remove puffs and waves from labels, print only if true
	* **noise_data_augmentation**: if true, apply some denoising/add noise randomly to the dataset movies as data augmentation
	* **sparks_type**: preprocessing applied to the spark labels, options are 'raw' or 'peaks'
3. NN architecture parameters

	* **nn_architecture**: used to select which network architecture to use, for the moment, the options are 'pablos_unet' and 'github_unet'
	* **unet_steps**: depth of the UNet architecture
	* **first_layer_channels**: number of output channels in the first convolutional layer
	* **temporal_reduction**: if true, process the input into a simple convolutional NN to reduce its temporal dimension, while increasing the number of channels of the input given to the UNet
	* **num_channels**: usually 1, it has a bigger value if e.g. using temporal reduction
	* **dilation**: if true, compute dilated convolution in the UNet model, not printed if not used
	* **border_mode**: possible values are 'same' and 'valid', if 'same' pad sampled input so that also border points are used during the convolution
	* **batch_normalization**: type of normalisation applied at the end of each block, for the moment possible values are 'batch' or 'none'
	* **initialize_weights**: if true, initialise network weights using a normal distribution
	* **attention**: ...


### Other parameters

These parameters are not important to be printed for comparison with the successive trainings.

🠊 They are loaded in the script _training.py_ from the configuration file directly where they are used

1. General parameters
	* **wandb_enable**: if true, log the training outputs and testing metrics on WandB
	* **training**: true to train the network
	* **testing**: true to run testing function at the end of training
2. Training parameters
	* **save_every**: number of iterations between model saving
	* **test_every**: number of iterations between testing
	* **print_every**: number of iterations between logging current training loss


### Currently not used

These parameters are currently unused, they are only present in the *config_....ini* files.

* **fixed_threshold**: detection threshold for test function, currently not used
* **t_detection_sparks**: threshold applied to sparks detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* **t_detection_puffs**: threshold applied to puffs detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* **t_detection_waves**: threshold applied to waves detection to compute metrics (currently not used since computing ouptput segmentation with argmax, hence its value is 'none' and it is not printed)
* **sparks_min_radius**: minimal radius below which spark predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
* **puffs_min_radius**: minimal radius below which puff predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
* **waves_min_radius**: minimal radius below which wave predictions are removed in the output segmentation (currently not used hence it value is 'none' and it is not printed)
