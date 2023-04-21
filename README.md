# ECG Data Classification using ConvLSTM

This project aims to classify ECG data with AAMI heart disease classes standard using a Convolutional LSTM deep learning model. It has been developed by Zibin ZHAO at the HSING Group.

## Overview

The code in this repository trains a ConvLSTM model on ECG data to classify different types of cardiac events, such as normal beats, left bundle branch block beats, right bundle branch block beats, and others. The model is trained on a dataset of ECG signals, which are preprocessed, normalized, and label-encoded before training.

The model architecture consists of three convolutional layers followed by a max-pooling layer and an LSTM layer. The output of the LSTM layer is fed into a fully connected layer, followed by a softmax layer for classification.

## Requirements

- Python 3.7 or higher
- NumPy
- Pandas
- scikit-learn
- PyTorch
- PyTorch Lightning
- TensorBoard (optional, for logging)

## Usage

1. Clone this repository to your local machine.
2. Ensure that you have the required Python packages installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

3. Prepare your ECG dataset in CSV format, with each row representing a single ECG signal. Save the dataset files as Concatenated_X.csv and Concatenated_y.csv in the data directory. (training dataset)
4. To retrain the model, run:

```bash
python main.py
```
This will train the model on the ECG dataset and save the trained model as trained_model.pth.

5. To use the trained model for predictions on new ECG data, resaple your ECG data to the shape (m, 3600) with m being the number of samples, and 3600 is the sampling frequency in the training data with 10 seconds duration for each sample, please refer to the predict.py script provided in the repository. (change to your own test data path)

## License
This project is released under the MIT License. 

### Supplementary Files
- Other model architectures are also provided for your interest including a 'convolution transformer' and also some variable using different deep learning framework (i.e.pytorch-lightning)
