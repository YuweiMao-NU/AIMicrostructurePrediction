import torch
import joblib
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train import EncoderDecoderRNN
from train import load_data
import json
import scipy
from train import data_normalized

def constrain(odfs):
    # Load properties from the MATLAB file
    mat = scipy.io.loadmat('../Copper_Properties.mat')
    p = mat['stiffness']
    q = mat['volumefraction']

    # Convert q to a numpy array if it is not already
    q = np.array(q, dtype=np.float64)

    # Initialize the output array with the same shape as odfs
    output_odf_list = np.zeros_like(odfs, dtype=np.float64)

    # Iterate over each row in odfs (9 rows in total)
    for i in range(odfs.shape[0]):
        odf = odfs[i]

        # Ensure all values are non-negative (like torch.maximum with 0)
        odf = np.maximum(odf, 0.0)

        # Calculate volume fraction
        volfrac = np.dot(q, odf)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        volfrac = volfrac + epsilon

        # Calculate constrained output (normalize by volume fraction)
        out_odf = odf / volfrac

        # Store the result in the output array
        output_odf_list[i] = out_odf

    return output_odf_list

# Define the same Dataset class that was used for training
class TimeSeriesDataset(Dataset):
    def __init__(self, x, target, param):
        self.x = x
        self.target = target
        self.param = param

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.target[index], self.param[index]

# Define the function to load the trained model
def load_model(model_path, input_size, hidden_size, num_layers, dropout, param_size):
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout, param_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

# Define the function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    output_list = []
    with torch.no_grad():
        for i, (input_data, target_data, param) in enumerate(test_loader):

            # Predict the future sequence
            output = model(input_data, param, iftrain=False, max_length=target_data.size(1))

            # Compute the loss
            loss = criterion(output, target_data)
            total_loss += loss.item() * input_data.size(0)

            output = output.numpy().reshape(-1, 76)
            # output = constrain(output)
            output_list.append(output)

    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')
    output_list = np.array(output_list)[0]
    # print(output_list.shape)
    return avg_loss, output_list

# Example usage
if __name__ == '__main__':

    batch_size = 1
    input_size = 76
    param_size = 5
    hidden_size = 128
    num_layers = 2
    dropout = 0.1
    sequence_length = 3  # Using the first H time steps
    future_length = 11 - sequence_length  # Predict the next F time steps
    num_epochs = 1000
    learning_rate = 1e-4
    p_teacher_forcing = 0.2
    model_path = 'autoencoder_new3.model'

    # Initialize the model
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout, param_size)

    with open("filenames.json", "r") as infile:
        filenames = json.load(infile)
    test_filenames = filenames['test_filenames']
    print(len(test_filenames))

    # Load the scaler from the file
    input_scaler = joblib.load('scaler.pkl')

    total_mse = 0
    for test_filename in test_filenames:
        test_input_data_list, test_param_list = load_data([test_filename])

        test_input_data_list, test_target_data_list = test_input_data_list[:, :sequence_length, :], test_input_data_list[:,sequence_length:,:]

        test_input_data_list, _ = data_normalized(test_input_data_list, scaler=input_scaler)

        test_input_data_list = torch.Tensor(test_input_data_list)
        test_target_data_list = torch.Tensor(test_target_data_list)
        test_param_list = torch.Tensor(test_param_list)

        testdata = TimeSeriesDataset(test_input_data_list, test_target_data_list, test_param_list)
        test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=False)

        # Load the trained model
        model = load_model(model_path, input_size, hidden_size, num_layers, dropout, param_size)

        # Define the loss criterion
        criterion = nn.MSELoss()

        # Test the model
        mse, output_list = test_model(model, test_loader, criterion)
        total_mse += mse

        [t1, t2, t3, t4, t5] = test_filename
        filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
        np.savetxt('results_new3/' + filename + '_predict.csv', output_list, delimiter=",")

    print(f'average mse is {total_mse/len(test_filenames)}')