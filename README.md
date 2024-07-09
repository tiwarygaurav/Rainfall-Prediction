# Maharashtra Rainfall Prediction

This repository contains a simple Artificial Neural Network (ANN) implementation using TensorFlow and Keras to forecast monthly rainfall in Maharashtra.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Dataset](#dataset)


## Overview
This project aims to forecast monthly rainfall in Maharashtra using an ANN model. The dataset used includes the year, month, and rainfall amount. The model is trained to predict the rainfall based on the given year and month.

## Installation
Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/tiwarygaurav/maharashtra-rainfall-prediction.git
    cd maharashtra-rainfall-prediction
    ```

2. **Create a new Conda environment** (optional but recommended):
    ```bash
    conda create --name tf_env python=3.8
    conda activate tf_env
    ```

3. **Install the required packages**:
    ```bash
    conda install tensorflow
    pip install numpy pandas scikit-learn
    ```

## Usage
1. **Prepare your dataset**:
    Ensure your dataset is in the format where one column is 'Year', one is 'Month', and one is 'Rainfall'. Place your dataset as a CSV file in the project directory.

2. **Run the script**:
    ```bash
    python rainfall_prediction.py
    ```

3. **Output**:
    The script will print the test loss and Mean Absolute Percentage Error (MAPE). It will also display the first few rows of the actual vs. predicted rainfall values.

## Model Description
The model is a simple ANN with the following architecture:
- Input layer: 2 neurons (Year and Month)
- Hidden layer 1: 32 neurons, ReLU activation
- Hidden layer 2: 16 neurons, ReLU activation
- Output layer: 1 neuron (Rainfall)

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function. It is trained for 100 epochs with a batch size of 10.

## Dataset
The dataset should include columns for 'Year', 'Month', and 'Rainfall'. Here is a sample dataset used for demonstration purposes:

```python
data = {
    'Year': np.tile(np.arange(2000, 2021), 12),
    'Month': np.repeat(np.arange(1, 13), 21),
    'Rainfall': np.random.uniform(50, 500, 252)  # Random data for example
}
df = pd.DataFrame(data)
