# Pet Classification

A deep learning project for binary classification of dogs vs. cats using TensorFlow and Keras.

## Overview

This project implements a convolutional neural network (CNN) to classify images as either dogs or cats. The notebook demonstrates multiple approaches using transfer learning (VGG16) and custom CNN architectures with data augmentation and normalization techniques.

## Dataset

- **Source**: Dogs vs. Cats dataset from Kaggle (salader/dogsvscats)
- **Format**: Image classification dataset with separate train and test directories
- **Image Size**: 256x256 pixels
- **Batch Size**: 32

## Features

### Data Processing
- Image dataset loading from directory with automatic label inference
- Pixel normalization (0-1 range)
- Data augmentation techniques:
  - Random horizontal flipping
  - Random rotation (0.1)
  - Random zoom (0.1)
  - Random contrast adjustment (0.1)

### Models

#### 1. Transfer Learning with VGG16
- Pre-trained VGG16 base (ImageNet weights)
- Fine-tuned classification head:
  - Flatten layer
  - Dense(256, relu)
  - Dense(1, sigmoid) for binary classification

#### 2. Custom CNN Architecture
- **Conv Layer 1**: 32 filters, 3x3 kernel, ReLU activation + BatchNormalization + MaxPooling
- **Conv Layer 2**: 64 filters, 3x3 kernel, ReLU activation + BatchNormalization + MaxPooling
- **Conv Layer 3**: 128 filters, 3x3 kernel, ReLU activation + BatchNormalization + MaxPooling
- **Dense Layers**: 
  - Dense(128, relu) with L2 regularization (0.01) + Dropout(0.1)
  - Dense(64, relu) with L2 regularization (0.01) + Dropout(0.1)
  - Dense(1, sigmoid) with L2 regularization (0.01)

## Requirements

- TensorFlow
- Keras
- kagglehub (for dataset download)
- Python 3.7+

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook pet_classification.ipynb
```

The notebook will:
1. Download the Dogs vs. Cats dataset from Kaggle
2. Load and preprocess the training and validation data
3. Apply data augmentation
4. Build and train the neural network models
5. Evaluate performance

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Data Augmentation**: Keras Sequential API
- **Regularization**: L2 regularization and Dropout
- **Image Processing**: Keras utilities for image dataset loading

## Notes

- The VGG16 base is frozen (non-trainable) for transfer learning approach
- L2 regularization (0.01) is applied to dense layers to prevent overfitting
- Dropout (0.1) is used in the custom CNN architecture
- BatchNormalization is applied after each convolutional layer
