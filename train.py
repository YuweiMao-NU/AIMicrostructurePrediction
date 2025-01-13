import numpy as np
import scipy.io
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import json
import joblib
import time
import pandas as pd

import torch
from torch import nn
# torch.set_default_tensor_type(torch.DoubleTensor)

class EncoderDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, param_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.param_size = param_size

        # Define the encoder LSTM
        self.encoder = nn.LSTM(input_size *2,
                               hidden_size,
                               num_layers,
                               dropout=dropout,
                               batch_first=True)

        # Define the decoder LSTM
        self.decoder = nn.LSTM(input_size,
                               hidden_size,
                               num_layers,
                               dropout=dropout,
                               batch_first=True)

        # Define the output layer
        self.param_embedding = nn.Linear(param_size, input_size)
        self.output = nn.Linear(hidden_size, input_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)

    def forward(self, x, processing_params, target=None, iftrain=True, max_length=9):
        batch_size = x.size(0)

        # Concatenate the processing parameters to each time step of the input sequence
        processing_params = self.param_embedding(processing_params)
        processing_params = processing_params.unsqueeze(1).repeat(1, x.size(1), 1)
        # print(x.size(), processing_params.size())
        encoder_input = torch.cat([x, processing_params], dim=2)

        # Encode the input sequence
        encoder_hidden = self.init_hidden(batch_size)
        # print(encoder_input.size())
        encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)

        # Prepare decoder input (starting with the last encoder output)
        decoder_input = x[:, -1:, :]  # Start with the last input from the initial sequence
        decoder_hidden = encoder_hidden

        # Initialize the output sequence
        output_seq = torch.zeros(batch_size, max_length, self.input_size)

        # Decode the output sequence
        for i in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            output = self.output(decoder_output.squeeze(1))
            # output = constrain(output)
            output_seq[:, i, :] = output

            # Use teacher forcing or the model's own predictions as the next input
            if iftrain and target is not None and np.random.random() < p_teacher_forcing:
                decoder_input = target[:, i:i+1, :]
            else:
                decoder_input = output.unsqueeze(1)

        return output_seq

# Define Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, x, target, param):
        self.x = x
        self.target = target
        self.param = param

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.target[index], self.param[index]


def random_split_param_filenames():
    interval = [0, 0.25, 0.5, 0.75, 1]
    filename_list = []
    for t1 in interval:
        for t2 in interval:
            for t3 in interval:
                for t4 in interval:
                    for t5 in interval:
                        filename = [t1, t2, t3, t4, t5]
                        filename_list.append(filename)

    train_filenames, test_filenames = train_test_split(filename_list, test_size=0.3, random_state=0)
    train_filenames, val_filenames = train_test_split(train_filenames, test_size=0.2, random_state=0)

    return train_filenames, val_filenames, test_filenames

def load_data(filename_list):
    inputs = []
    param_list = []
    for filename in filename_list:
        [t1, t2, t3, t4, t5] = filename
        # param_rep = onehot_encode(filename).reshape(1, -1)
        param_rep = np.array([t1, t2, t3, t4, t5]).reshape(1, -1)
        # filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
        filename = str(t1) + '_' + str(t2) + '_' + str(t3) + '_' + str(t4) + '_' + str(t5)
        odf = np.loadtxt('ODF_new/' + filename + '.csv', delimiter=',').transpose()
        # print(tmp.shape)
        inputs.append(odf)
        param_list.append(param_rep[0])
    inputs, param_list = np.array(inputs), np.array(param_list)


    return inputs, param_list

def constrain(odfs):
    mat = scipy.io.loadmat('../Copper_Properties.mat')
    p = mat['stiffness']
    q = mat['volumefraction']
    q = torch.tensor(q, dtype=torch.double)  # convert to PyTorch tensor

    output_odf_list = torch.zeros_like(odfs)

    for i in range(odfs.size()[0]):
        odf = odfs[i]
        odf = torch.maximum(odf, torch.tensor(0.0))
        odf = odf.type(torch.DoubleTensor)  # convert to PyTorch tensor
        volfrac = torch.matmul(q, odf) + 1e-8
        out_odf = odf/volfrac
        output_odf_list[i] = out_odf  # convert back to NumPy array

    return output_odf_list

def data_normalized(target_data, scaler=None):
    num_samples, seq_length, num_features = target_data.shape
    reshaped_data = target_data.reshape(num_samples * seq_length, num_features)

    # Step 2: Normalize using StandardScaler
    if not scaler:
        scaler = StandardScaler()
        reshaped_data_normalized = scaler.fit_transform(reshaped_data)
    else:
        reshaped_data_normalized = scaler.transform(reshaped_data)
    # Step 3: Reshape back to the original shape
    target_data_normalized = reshaped_data_normalized.reshape(num_samples, seq_length, num_features)
    return target_data_normalized, scaler

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_filenames, val_filenames, test_filenames = random_split_param_filenames()
    print('train file number: ', len(train_filenames), 'val file number:', len(val_filenames), 'test file number: ', len(test_filenames))
    filenames = {"train_filenames":train_filenames, "val_filenames":val_filenames, "test_filenames":test_filenames}
    filenames = json.dumps(filenames)
    with open("filenames.json", "w") as outfile:
        outfile.write(filenames)

    batch_size = 16
    input_size = 76
    param_size = 5
    hidden_size = 128
    num_layers = 2
    dropout = 0.1
    sequence_length = 3  # Using the first H time steps
    future_length = 11 - sequence_length   # Predict the next F time steps
    num_epochs = 1000
    learning_rate = 1e-4
    p_teacher_forcing = 0.2

    model_name = 'autoencoder_new3.model'

    # Initialize the model
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout, param_size)
    # model = model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    full_data_list, param_list = load_data(train_filenames)
    input_data_list = full_data_list[:, :sequence_length, :]  # Shape: (100, 2, 76)

    # target_data_list takes the next 9 time steps
    target_data_list = full_data_list[:, sequence_length:, :]  # Shape: (100, 9, 76)
    print(input_data_list.shape, param_list.shape)
    #
    input_data_list, input_scaler = data_normalized(input_data_list, scaler=None)
    joblib.dump(input_scaler, 'scaler.pkl')

    # target_data_list = target_data_list.to(device)
    input_data_list = torch.Tensor(input_data_list)
    target_data_list = torch.Tensor(target_data_list)
    param_list = torch.Tensor(param_list)
    traindata = TimeSeriesDataset(input_data_list, target_data_list, param_list)
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

    # load val data
    val_input_data_list, val_param_list = load_data(val_filenames)

    val_input_data_list, val_target_data_list = val_input_data_list[:, :sequence_length, :], val_input_data_list[:, sequence_length:, :] # Shape: (100, 2, 76)

    val_input_data_list, _ = data_normalized(val_input_data_list, scaler=input_scaler)

    val_input_data_list = torch.Tensor(val_input_data_list)
    val_target_data_list = torch.Tensor(val_target_data_list)
    val_param_list = torch.Tensor(val_param_list)

    valdata = TimeSeriesDataset(val_input_data_list, val_target_data_list, val_param_list)
    val_loader = DataLoader(valdata, batch_size=batch_size, shuffle=False)


    # Initialize the model
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout, param_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for i, (input_data, target_data, param) in enumerate(train_loader):
            optimizer.zero_grad()
            input_data, target_data, param = input_data, target_data, param

            output = model(input_data, param, target_data, iftrain=True, max_length=future_length)

            loss = criterion(output, target_data)
            if torch.isnan(loss).any():
                print(f"Skipping step {i} due to NaN loss")
                continue

            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item() * input_data.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}')

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (input_data, target_data, param) in enumerate(val_loader):
                input_data, target_data, param = input_data, target_data, param

                output = model(input_data, param, iftrain=False, max_length=future_length)
                val_loss = criterion(output, target_data)
                total_val_loss += val_loss.item() * input_data.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss}')

    print("Training complete.")
    torch.save(model.state_dict(), model_name)