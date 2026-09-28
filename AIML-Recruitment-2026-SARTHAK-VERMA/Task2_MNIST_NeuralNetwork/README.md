# MNIST Neural Network Task

## Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset using TensorFlow/Keras. The model achieves high accuracy through convolutional feature extraction, pooling layers, and dense classification layers.

## Dataset
- **Source**: MNIST dataset (built-in to TensorFlow/Keras)
- **Size**: 70,000 grayscale images (28×28 pixels)
  - 60,000 training images
  - 10,000 test images
- **Classes**: 10 digits (0-9)

## Methodology
### 1. Data Loading & Exploration
- Loaded dataset using `tf.keras.datasets.mnist.load_data()`
- Examined shape, pixel value range (0-255), and class distribution
- Visualized sample training images

### 2. Data Preprocessing
- **Normalization**: Pixel values scaled to [0,1] range by dividing by 255.0
- **Reshaping**: Added channel dimension for CNN compatibility → shape: (samples, 28, 28, 1)
- **One-hot Encoding**: Converted integer labels to categorical format using `tf.keras.utils.to_categorical()`

### 3. Model Architecture
Built a CNN with:
- **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D Layer 3**: 64 filters, 3×3 kernel, ReLU activation
- **Flatten Layer**: Converted 3D feature maps to 1D vector
- **Dense Layer**: 64 units, ReLU activation
- **Dropout Layer**: 50% dropout rate for regularization
- **Output Layer**: 10 units, softmax activation (probabilities per digit)

### 4. Model Training
- **Optimizer**: Adam
- **Loss Function**: Categorical crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 128
- **Epochs**: 10
- **Validation Split**: 10% of training data held out for validation

### 5. Model Evaluation & Visualization
- **Test Set Evaluation**: Reported accuracy and loss
- **Confusion Matrix**: Heatmap showing prediction performance per digit
- **Classification Report**: Precision, recall, F1-score for each class
- **Prediction Visualization**: Grid of sample predictions (correct=green, incorrect=red)

## Files
- `notebooks/mnist_neural_network.ipynb`: Complete Jupyter notebook with all 6 cells
- `data/`: Directory (MNIST dataset auto-downloaded by TensorFlow)
- `README.md`: This file
- `mnist_model.h5`: Saved trained model (generated during execution)

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for confusion matrix and classification report)

## How to Run
1. Install requirements: `pip install tensorflow matplotlib numpy seaborn scikit-learn`
2. Open `mnist_neural_network.ipynb` in Jupyter Notebook or VS Code
3. Run all cells in sequence (Cell 0 → Cell 5)
4. The model will train, evaluate, and save as `mnist_model.h5`

## Key Results
- **Test Accuracy**: ~99% (exact value visible in notebook output)
- **Model Parameters**: ~1.2M trainable parameters
- **Training Time**: ~2-5 minutes on CPU
- **Architecture**: Effective CNN design suitable for digit recognition

## Notes
- The model uses a simple but effective CNN architecture appropriate for MNIST
- Dropout regularization helps prevent overfitting
- Validation split during training monitors for overfitting
- Saved model (`mnist_model.h5`) can be reloaded for future predictions