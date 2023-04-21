"Predict your own ECG classes using a pretrained model by Zibin ZHAO @HSING Group"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from model import ConvLSTM  


def load_ecg_data(filepath):
    ecg_data = pd.read_csv(filepath, header=None).values
    print(f"Loaded ECG data shape: {ecg_data.shape}") 
    return ecg_data

def preprocess_ecg_data(ecg_data):
    scaler = StandardScaler()
    ecg_data_normalized = scaler.fit_transform(ecg_data)
    return ecg_data_normalized

def load_trained_model(model_path):
    model = ConvLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_ecg_class(model, ecg_data):
    inputs = torch.tensor(ecg_data, dtype=torch.float32)
    inputs = inputs.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(inputs)
    probabilities, predicted = torch.max(output, 1)
    return predicted.item(), probabilities


if __name__ == "__main__":
    # Load your ECG data
    ecg_data_filepath = "./data/segmented_data.csv"  # Replace with the path to your ECG data
    ecg_data = load_ecg_data(ecg_data_filepath)

    # Preprocess ECG data
    ecg_data_normalized = preprocess_ecg_data(ecg_data)
    print(ecg_data_normalized.shape)

    # Load the trained model
    model_path = "trained_model.pth"
    model = load_trained_model(model_path)

    # Predict the ECG class
    for i in range(ecg_data_normalized.shape[0]):
        predicted_class, probabilities = predict_ecg_class(model, ecg_data_normalized[i, :])

        # Print the results
        print(f"Predicted class: {predicted_class}")
        print(f"Probabilities: {probabilities}")
