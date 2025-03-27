# Siamese Neural Network for MNIST Digit Comparison

This repository contains an implementation of a Siamese Neural Network for comparing pairs of MNIST handwritten digits. The model is trained to determine whether two digit images represent the same digit or different digits.

## Project Overview

Siamese networks are neural network architectures that contain two identical subnetworks with shared weights. These networks learn to find the similarity between inputs by comparing their feature representations. In this implementation, we use a Siamese network to compare MNIST digit images and determine if they represent the same digit. 

## Features

- PyTorch implementation of a Siamese Neural Network
- Data preprocessing for creating pairs from the MNIST dataset
- Training and evaluation scripts
- Visualization tools for examining model predictions
- Alternative Keras implementation in Jupyter notebook format

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd siamese

# Install required packages
pip install torch torchvision matplotlib numpy
```

## Project Structure

- `main.py` - Main training script
- `test.py` - Testing and visualization utilities
- `pre_process.py` - Data preprocessing and dataset creation
- `net.py` - Siamese network architecture definition
- `Siamese_keras.ipynb` - Alternative implementation using Keras

## Usage

### Training the Model

```bash
python main.py
```

This will:
- Download the MNIST dataset (if not already present)
- Create pairs of similar and dissimilar digit images
- Train the Siamese network for the specified number of epochs
- Save the trained model to `siamese_model.pth`

### Testing the Model

```bash
python test.py
```

This will:
- Load the saved model
- Evaluate its performance on the test dataset
- Visualize example predictions with `plot_checker` function

## Implementation Details

The Siamese network processes pairs of MNIST digit images through identical neural networks with shared weights. The model then computes the similarity between the resulting feature vectors to determine if the input images represent the same digit.

Key components:
- `SiameseNet`: The neural network architecture with shared weights
- `SiamenseDataset`: Custom dataset class for handling image pairs
- `make_pairs`: Function to create pairs of same/different digit images
- `BCEWithLogitsLoss`: Binary cross-entropy loss for training

## Examples

After running the test script, visualizations will show pairs of digit images along with the model's prediction of whether they represent the same digit or different digits.

## License

[Add your license information here]
