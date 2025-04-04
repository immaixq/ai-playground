# Finetuning CLIP Image Encoder with Supervised Contrastive Learning

This document provides instructions on how to set up and run the experiment for finetuning the image encoder of a CLIP (Contrastive Language-Image Pre-training) model using a supervised contrastive learning approach. The goal is to improve the model's ability to classify images of cats and dogs.

## Repository Structure

This experiment's code is organized within the following directory structure:

```python

├── configs
│   └── clip_supercon_config.py  # Configuration file for training
├── datasets
│   └── clip_supcon_dataset.py  # Dataset loading and preprocessing logic
├── main.py                     # Main script to run the training process
├── models
│   └── supcon_clip.py           # Script for evaluating the finetuned model
├── output_supcon               # Directory to store training outputs (logs, checkpoints)
│   └── training_supcon.log
└── utils
└── helper.py               # Utility functions

```

## Prerequisites

Before you begin, ensure you have the following installed:

```bash
conda env create -f environment.yml
```

Install CLIP from official repository

```python
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation
Replace the `/data` directory to your custom data for finetuning with the following structure:

```python
.
├── {class1}
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
└── {class2}
    ├── 1.jpg
    ├── 2.jpg
    └── 3.jpg

```

## Configuration

The training parameters are defined in the `configs/clip_supercon_config.py` file. You can open this file and adjust the following parameters as needed:

* **`model_name`**: The name of the pre-trained CLIP model to use (e.g., `"openai/clip-vit-base-patch32"`).
* **`data_path`**: This should be set to the path of your data directory (`"../../data/clip"` relative to the `main.py` script).
* **`batch_size`**: The number of images processed in each training batch. Adjust based on your GPU memory.
* **`learning_rate`**: The learning rate for the optimizer.
* **`num_epochs`**: The total number of training epochs.
* **`image_size`**: The input image size expected by the CLIP model.
* **`projection_dim`**: The dimension of the projection head (if used).
* **`temperature`**: The temperature parameter for the contrastive loss.
* **`output_dir`**: The directory where training outputs will be saved (defaults to `"output_supcon"`).
* **`log_interval`**: How often to log training progress (in steps).
* **`save_interval`**: How often to save model checkpoints (in epochs).

Review these parameters and modify them according to your needs and available resources.

## Running the Training

To start the finetuning process, navigate to the `src/clip_finetuning` directory in your terminal (if you're not already there) and run the `main.py` script:

```bash
python main.py