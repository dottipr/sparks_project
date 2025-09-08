# 🔬 Detection and Classification of Local Ca²⁺ Release Events in Cardiomyocytes with 3D U-Net

This repository contains the code, trained models, and resources developed for the detection and classification of local Ca²⁺ release events in cardiomyocytes using a **3D U-Net neural network**.

📄 The work is described in our publication:  
> **A deep learning-based approach for efficient detection and classification of local Ca²⁺ release events in Full-Frame confocal imaging**, *Cell Calcium, 2024* ([DOI link](https://doi.org/10.1016/j.ceca.2024.102893))  

🗂️ The project also includes the open-source dataset ([Zenodo link](https://doi.org/10.5281/zenodo.10391727)) for reproducibility. This dataset includes the annotations of Ca²⁺ sparks, Ca²⁺ puffs, and Ca²⁺ waves.

![Graphical Abstract](https://ars.els-cdn.com/content/image/1-s2.0-S0143416024000514-ga1_lrg.jpg)

---

## 🚀 Features
- End-to-end pipeline for **training and inference** with 3D U-Net on calcium imaging movies.  
- Tools for **data preprocessing, annotation generation, and visualization**.  
- Custom **loss functions** and evaluation metrics.  
- Pre-trained model checkpoints available for inference.  
- Example **Jupyter notebooks** for interactive experimentation.  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/dottipr/sparks_project.git
cd sparks_project
```

Create and activate an environment (conda or venv):
```bash
conda env create -f requirements/environment.yml
conda activate sparks_project
```

Alternatively:
```bash
pip install -r requirements/requirements.txt
```

---

## 🏃 Usage

### Training

Training is configured through a `.ini` file located in **`config_files/`**.  
This file defines the experiment name, dataset parameters, model architecture, and training settings.  

To launch training, run:

```bash
python run_training.py config_files/example_config.ini
```

Or, for debugging and step-by-step runs, use the interactive notebook:

```jupyter
notebooks/training (interactive).ipynb
```

---

#### Configuration File

Each training run is controlled by a `.ini` configuration file with the following sections:

- **[general]** → experiment metadata, WandB logging, training/validation switches  
- **[training]** → run name, loss function, learning rate, number of epochs, checkpoint saving frequency  
- **[dataset]** → dataset location, batch size, preprocessing (normalization, background removal, etc.)  
- **[network]** → U-Net architecture, depth, channels, upsampling mode, normalization layers  
- **[inference]** → parameters for validation/inference during training  



---

#### Notes about parameters
- `run_name` defines the folder where models and logs are saved.  
- `criterion` can be `lovasz_softmax`, `focal_loss`, `nll_loss`, etc.  
- `dataset_size` can be `full` or `minimal` (useful for debugging).  
- `remove_background` and `sparks_type` allow experimenting with different preprocessing strategies.  
- `nn_architecture` can be swapped to test different U-Net variants.  

---

### Inference

To run inference, edit the parameters at the top of **`run_inference.py`**:

- `run_name` → name of the training run (e.g. `"final_model"`)  
- `config_filename` → the `.ini` config file used during training (must be in `config_files/`)  
- `load_epoch` → epoch number of the model checkpoint to load (e.g. `100000`)  
- `custom_ids` → optionally specify which sample IDs to run (e.g. `["05"]`)  
- `testing` → set `True` to compute processed outputs and metrics, `False` for raw predictions only  

Then launch inference:

```bash
python run_inference.py
```

Predictions will be saved in:

```
evaluation/inference_script/<run_name>/
```

with filenames in the format:

```
{training_name}_{epoch}_{video_id}_{class}.tif
```

For interactive exploration and debugging, you can also use the notebook:

```jupyter
notebooks/inference (interactive).ipynb
```

---

## 📂 Repository Structure

```bash
sparks_project/
├── config_files/              # Training & inference config (.ini) files
├── data/
│   ├── sparks_dataset/        # Dataset samples (.tif movies, labels, masks)
│   ├── data_processing_tools.py
│   ├── datasets.py            # Dataset class
│   └── generate_unet_annotations.py
├── evaluation/
│   ├── script1_output/        # Example evaluation results
│   ├── script2_output/
│   └── metric_tools.py
├── models/
│   ├── saved_models/          # Trained models + checkpoints
│   ├── nnUNet/                # nnUNet implementation (experimental)
│   ├── UNet/                  # Final U-Net implementation
│   ├── unetOpenAI/            # OpenAI U-Net (experimental)
│   ├── architectures.py       # U-Net model variations (experimental)
│   ├── new_unet.py            # Alternative U-Net implementation (experimental)
│   ├── saved_models.zip.001
│   └── saved_models.zip.002
├── notebooks/
│   ├── inference (interactive).ipynb
│   ├── matlab inference.ipynb
│   ├── plot and analyze detected events.ipynb
│   ├── training (interactive).ipynb
│   └── save processed movies on disk.ipynb
├── raw_notebooks/             # Experimental notebooks
├── requirements/              # Environment & dependency specs
├── utils/
│   ├── LovaszSoftmax/
│   ├── custom_losses.py
│   ├── in_out_tools.py
│   ├── training_inference_tools.py
│   ├── training_script_utils.py
│   └── visualization_tools.py
├── config.py                  # Global + training-specific config classes
├── run_inference.py           # Run inference with a trained model
└── run_training.py            # Train a model from a config file
```

#### Note

Unzip and combine `models/saved_models.zip.001` and `models/saved_models.zip.002` to get the trained final model used in our publication.

---

## 📊 Dataset

The dataset consists of calcium imaging movies (`NN_video.tif`) with associated labels:
- **Class labels** (`NN_class_label.tif`) → segmentation of local calcium release events (Ca²⁺ sparks, Ca²⁺ puffs, and Ca²⁺ waves).  
- **Event masks** (`NN_event_label.tif`) → instance masks with individual events.  

More details and downloads: [Zenodo link](https://doi.org/10.5281/zenodo.10391727)

---

## 🤝 Contributing

Issues, pull requests, and suggestions are welcome!  
If you use this repository, please **cite our paper**:  

```bibtex
@article{dotti2024,
  title={A deep learning-based approach for efficient detection and classification of local Ca²⁺ release events in Full-Frame confocal imaging},
  author={Prisca Dotti and Miguel Fernandez-Tenorio and Radoslav Janicek and Pablo Márquez-Neila and Marcel Wullschleger and Raphael Sznitman and Marcel Egger},
  journal={Cell Calcium},
  year={2024},
  doi={https://doi.org/10.1016/j.ceca.2024.102893}
}
```

---

## 📧 Contact

For questions, please reach out:  
👩‍💻 Prisca Dotti – prisca.dotti@outlook.com
