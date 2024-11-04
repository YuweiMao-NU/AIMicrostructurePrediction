import numpy as np
import scipy.io
import torch
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

import torch
from torch import nn
# torch.set_default_tensor_type(torch.DoubleTensor)

class EncoderDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define the encoder LSTM
        self.encoder = nn.LSTM(input_size*(sequence_length+1)+processing_params_size,
                                   hidden_size,
                                   num_layers,
                                   dropout=dropout,
                                   batch_first=True)
        # self.encoder = nn.Linear(input_size*2, hidden_size)

        # Define the decoder LSTM
        self.decoder = nn.LSTM(input_size*2,
                                   hidden_size,
                                   num_layers,
                                   dropout=dropout,
                                   batch_first=True)

        # Define the output layer
        self.param_embedding = nn.Linear(processing_params_size, input_size)
        self.output = nn.Linear(hidden_size, input_size)

    def init_hidden(self, batch_size):
        # initialize hidden state and cell state with zeros
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # print(h_0.dtype)
        return (h_0, c_0)

    def forward(self, x, processing_params, target, iftrain=True, max_length=10):
        x = x.unsqueeze(1)
        batch_size = x.size(0)
        encoder_hidden = self.init_hidden(batch_size)
        decoder_hidden = self.init_hidden((batch_size))

        # Concatenate the processing parameters to x[0]
        processing_params = self.param_embedding(processing_params)
        # print(processing_params.size())

        # add param into x
        encoder_processing_params = processing_params.unsqueeze(1).repeat(1, x.size(1), 1)
        encoder_input = torch.cat([x, encoder_processing_params], dim=2)
        # encoder_input = torch.add(x, encoder_processing_params)

        # Encode the input sequence
        encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
        # print(encoder_output.size())

        decoder_processing_params = processing_params.unsqueeze(1).repeat(1, 1, 1)
        # print('param: ', processing_params.size())
        decoder_input = torch.cat([target[:, :1, :], decoder_processing_params], dim=2)
        # decoder_input = torch.add(x[:, :1, :], decoder_processing_params)
        # print(decoder_input.size())

        # Initialize the output sequence
        output_seq = torch.zeros(batch_size, 11-sequence_length, self.input_size)

        # Decode the output sequence
        if iftrain==True:
            # Teacher-forcing during training
            decoder_hidden = encoder_hidden
            for i in range(11-sequence_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                output = self.output(decoder_output.squeeze(1))
                output_seq[:, i, :] = output
                # print('target: ',(target[:, i-1, :].unsqueeze(1)).size())
                if np.random.random() < p_teacher_forcing:
                    decoder_input = torch.cat((target[:, i, :].unsqueeze(1), decoder_processing_params), dim=2)
                    # decoder_input = torch.add(target[:, i, :].unsqueeze(1), decoder_processing_params)
                else:

                    # output = output.unsqueeze(1)
                    # print(output.size())
                    output = constrain(output)
                    decoder_input = torch.cat([output.unsqueeze(1), decoder_processing_params], dim=2)
                    # decoder_input = torch.add(output.unsqueeze(1), decoder_processing_params)
        else:
            # Generate output sequence during testing
            decoder_hidden = encoder_hidden
            for i in range(11-sequence_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                output = self.output(decoder_output.squeeze(1))
                # output = output.unsqueeze(1)
                output = constrain(output)
                output_seq[:, i, :] = output
                decoder_input = torch.cat([output.unsqueeze(1), decoder_processing_params], dim=2)
                # decoder_input = torch.add(output.unsqueeze(1), decoder_processing_params)

        return output_seq, decoder_hidden


def train_model(model, train_loader, criterion, optimizer, batch_size, num_epochs):
    min_loss = float('inf')
    best_model = model
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for i, (input_data, target_data, param) in enumerate(train_loader):
            optimizer.zero_grad()
            # print(input_data.dtype, target_data.dtype, param.dtype)
            # input_data = torch.FloatTensor(input_data)
            # target_data = torch.FloatTensor(target_data)
            # param = torch.FloatTensor(param)
            # print(input_data.size(), target_data.size(), param.size())

            output, decoder_hidden = model(input_data, param, target_data, iftrain=True, max_length=10)
            loss = criterion(output, target_data[:, sequence_length:, :])
            total_loss += loss.item() * input_data.size(0)
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader.dataset)
        print('Epoch [{}], train Loss: {}'.format(epoch, avg_loss))
        ave_loss = test_model(model, val_loader, criterion, batch_size)
        print('test loss: ', ave_loss)
        if ave_loss < min_loss:
            min_loss = ave_loss
            best_model = model
            torch.save(model.state_dict(), model_name)
    return best_model

def test_model(model, test_loader, criterion, batch_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (input_data, target_data, param) in enumerate(test_loader):
            output, decoder_hidden = model(input_data, param, target_data, iftrain=False, max_length=10)
            loss = criterion(output, target_data[:, sequence_length:, :])
            total_loss += loss.item() * input_data.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

def predict_model(model, test_loader, criterion):
    model.eval()

    total_mse = 0
    total_mape = 0
    output_list = []
    with torch.no_grad():
        for i, (input_data, target_data, param) in enumerate(test_loader):
            output, decoder_hidden = model(input_data, param, target_data, iftrain=False, max_length=10)
            # output = constrain(output)
            output, target_data = output.cpu().numpy().reshape(-1, 76), \
                                 target_data.cpu()[:, sequence_length:, :].numpy().reshape(-1, 76)
            # print(output.shape, target_data.shape)
            loss = mean_squared_error(output, target_data)
            # mape = mean_absolute_percentage_error(y_true=target, y_pred=output, multioutput='raw_values').reshape(1, -1)
            # print(mape.shape, output.shape, target.shape)
            # mape[np.where(target == 0)] = 0
            output_list.append(output)
            total_mse += loss
            # total_mape += np.average(mape, axis=0)

    output_list = np.array(output_list)[0]
    avg_mse = total_mse / (len(test_loader.dataset))
    avg_mape = total_mape / (len(test_loader.dataset))
    return output_list, avg_mse, avg_mape

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
    outputs = []
    param_list = []
    for filename in filename_list:
        [t1, t2, t3, t4, t5] = filename
        # param_rep = onehot_encode(filename).reshape(1, -1)
        param_rep = np.array([t1, t2, t3, t4, t5]).reshape(1, -1)
        filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
        odf = np.loadtxt('../ODF/' + filename + 'ODF.csv', delimiter=',').transpose()
        tmp = np.concatenate((odf[:sequence_length].reshape(1, -1), param_rep), axis=1)
        # print(tmp.shape)
        inputs.append(tmp[0])
        outputs.append(odf)
        param_list.append(param_rep[0])
    return np.array(inputs), np.array(outputs), np.array(param_list)


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
        volfrac = torch.matmul(q, odf)
        out_odf = odf/volfrac
        output_odf_list[i] = out_odf  # convert back to NumPy array

    return output_odf_list

class TimeSeriesDataset(Dataset):
    def __init__(self, x, target, param):
        self.x = x
        self.target = target
        self.param = param

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.target[index], self.param[index]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_filenames, val_filenames, test_filenames = random_split_param_filenames()
    print('train file number: ', len(train_filenames), 'val file number:', len(val_filenames), 'test file number: ', len(test_filenames))
    filenames = {"train_filenames":train_filenames, "val_filenames":val_filenames, "test_filenames":test_filenames}
    filenames = json.dumps(filenames)
    with open("filenames.json", "w") as outfile:
        outfile.write(filenames)

    input_size = 76  # size of each ODF vector
    hidden_size = 128  # number of hidden units in each layer
    num_layers = 2  # number of layers in each encoder and decoder LSTM
    dropout = 0.1  # dropout rate to apply to each LSTM layer
    sequence_length = 2
    processing_params_size = 5
    batch_size = 4
    num_epochs = 100
    learning_rate = 0.01
    log_interval = 10
    alpha = 0.1
    p_teacher_forcing = 0.2
    model_name = 'autoencoder_v1.model'
    criterion = nn.MSELoss()

    # Initialize the model
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout)
    # model = model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    input_data_list, target_data_list, param_list = load_data(train_filenames)
    print(input_data_list.shape, target_data_list.shape, param_list.shape)
    input_scaler = StandardScaler()
    input_data_list = input_scaler.fit_transform(input_data_list)
    # input_data_list = input_data_list.to(device)
    #
    # target_data_list = target_data_list.to(device)
    input_data_list = torch.Tensor(input_data_list)
    target_data_list = torch.Tensor(target_data_list)
    param_list = torch.Tensor(param_list)
    traindata = TimeSeriesDataset(input_data_list, target_data_list, param_list)
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)

    # load val data
    val_input_data_list, val_target_data_list, val_param_list = load_data(val_filenames)
    # print('test input size:', test_input_data_list.size(), 'target size: ', test_target_data_list.size(),
    #       'exogenous feature size: ', test_exogenous_input_list.size())

    # test_input_data_list = test_input_data_list.to(device)
    val_input_data_list = input_scaler.transform(val_input_data_list)
    # test_target_data_list = test_target_data_list.to(device)

    val_input_data_list = torch.Tensor(val_input_data_list)
    val_target_data_list = torch.Tensor(val_target_data_list)
    val_param_list = torch.Tensor(val_param_list)

    valdata = TimeSeriesDataset(val_input_data_list, val_target_data_list, val_param_list)
    val_loader = DataLoader(valdata, batch_size=batch_size, shuffle=False)

    # Train and save the model
    model = train_model(model, train_loader, criterion, optimizer, batch_size, num_epochs)
    # model = model.to(torch.device('cpu'))
    torch.save(model.state_dict(), model_name)

    #load model
    model = EncoderDecoderRNN(input_size, hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(model_name))
    # model = model.to(device)

    # Predict the next Fp using predicted Fp
    total_mse = 0
    total_mape = 0
    for test_filename in test_filenames:
        st = time.time()
        test_input_data_list, test_target_data_list, test_param_list = load_data([test_filename])
        # print('test input size:', test_input_data_list.size(), 'target size: ', test_target_data_list.size(),
        #       'exogenous feature size: ', test_exogenous_input_list.size())
        # test_input_data_list = test_input_data_list.to(device)
        test_input_data_list = input_scaler.transform(test_input_data_list)
        # test_target_data_list = test_target_data_list.to(device)

        test_input_data_list = torch.Tensor(test_input_data_list)
        test_target_data_list = torch.Tensor(test_target_data_list)
        test_param_list = torch.Tensor(test_param_list)

        testdata = TimeSeriesDataset(test_input_data_list, test_target_data_list, test_param_list)
        test_loader = DataLoader(testdata, batch_size=1, shuffle=False)

        output_list, mse, mape = predict_model(model, test_loader, criterion)
        # print(output_list.shape)
        print('time: ', time.time()-st)
        print(test_filename, ', mse: {}, mape: {}'.format(mse, mape))
        total_mse += mse
        total_mape += mape

        [t1, t2, t3, t4, t5] = test_filename
        filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
        np.savetxt('results_auencoder4/' + filename + '_predict.csv', output_list, delimiter=",")

    print('Test average mse: {}, mape: {}'.format(total_mse/len(test_filenames),
                                                  total_mape/len(test_filenames)))