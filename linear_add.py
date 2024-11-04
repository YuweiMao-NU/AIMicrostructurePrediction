import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import json
import time

def calS(C_array):
    S_array = np.zeros_like(C_array)
    for i in range(C_array.shape[0]):
        C = C_array[i]
        C = C.reshape(6, 6)
        S = np.linalg.inv(C)
        S = S.reshape(1, -1)[0]
        S_array[i] = S

    return S_array

def compare(true_array, pred_array):
    mape_list = []
    for i in range(true_array.shape[0]):
        true = true_array[i].reshape(1, -1)
        predict = pred_array[i].reshape(1, -1)

        mape = mean_absolute_percentage_error(true, predict, multioutput='raw_values').reshape(1, -1)
        # print(mape)
        mape[np.where(true == 0)] = 0
        # print(mape)
        mape_list.append(mape[0])
    mape_list = np.array(mape_list)
    # print(mape_list.shape)
    mape = np.average(mape_list, axis=0)
    return mape

def added_linear(path1, path2):
    with open("filenames.json", "r") as infile:
        filenames = json.load(infile)
    test_filenames = filenames['test_filenames']
    print(len(test_filenames))

    mape_predict1_list = []
    mape_predict2_list = []
    mape_predict2_S_list = []

    for filename in test_filenames:

        [t1, t2, t3, t4, t5] = filename
        filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
        true = np.loadtxt('results_C/' + filename + 'true_C.csv', delimiter=",")
        true_S = np.loadtxt('results_S/' + filename + 'true_S.csv', delimiter=",")
        predict = np.loadtxt(path1 + '/' + filename + 'predict_C.csv', delimiter=",")
        predict_S = np.loadtxt(path2 + '/' + filename + 'predict_S.csv', delimiter=",")

        # print(true.shape, predict.shape)
        mape_predict1 = compare(true[sequence_length:], predict)
        mape_predict1_list.append(mape_predict1)

        dif_predict = (true[0] + (true[1] - true[0]) * 2) - predict[0]

        # print(dif_predict, dif_predict_auto)
        predict = predict + dif_predict

        mape_predict2 = compare(true[sequence_length:], predict)
        mape_predict2_list.append(mape_predict2)

        # cal S
        dif_predict_S = (true_S[0] + (true_S[1] - true_S[0]) * 2) - predict_S[0]
        predict_S += dif_predict_S
        # print(predict_S)
        mape_predict2_S = compare(true_S[sequence_length:], predict_S)
        mape_predict2_S_list.append(mape_predict2_S)

    mape_predict1_list = np.array(mape_predict1_list)
    ave_mape_predict1 = np.average(mape_predict1_list, axis=0)

    mape_predict2_list = np.array(mape_predict2_list)
    array = np.average(mape_predict2_list, axis=1)
    print(array.shape, np.min(array), np.max(array), np.median(array))

    min_index = np.argmin(array)
    max_index = np.argmax(array)
    median_value = np.median(array)
    median_index = (np.abs(array - median_value)).argmin()
    print(min_index, max_index, median_index)
    print(test_filenames[min_index], test_filenames[max_index], test_filenames[median_index])

    ave_mape_predict2 = np.average(mape_predict2_list, axis=0)
    mape_predict2_S_list = np.array(mape_predict2_S_list)
    ave_mape_predict2_S_list = np.average(mape_predict2_S_list, axis=0)

    print(np.average(ave_mape_predict1), np.average(ave_mape_predict2), np.average(ave_mape_predict2_S_list))

    # print(ave_mape_predict_auto2)
    np.savetxt(path1 + '_dif_C2.csv', ave_mape_predict2.reshape(6, 6), delimiter=',')
    np.savetxt(path1 + '_dif_S2.csv', ave_mape_predict2_S_list.reshape(6, 6), delimiter=',')

def plot_com(path, filename, dim=0):
    true = np.loadtxt('results_C/' + filename + 'true_C.csv', delimiter=",")[:, dim]
    predict_auto = np.loadtxt(path + '/' + filename + 'predict_C.csv', delimiter=",")[:, dim]

    plt.figure(0)
    x = list(range(11))
    plt.plot(x[sequence_length - 1:], predict_auto, label='seq2seq', color='blue')

    st = time.time()
    # dif_predict = (true[0] + (true[1] - true[0])*2)-predict[2]
    dif_predict_auto = (true[0] + (true[1] - true[0])*2)-predict_auto[0]
    # dif_predict_auto2 = (true[0] + (true[1] - true[0]) * 2) - predict_auto2[2]

    # print(dif_predict, dif_predict_auto, dif_predict_auto2)
    # predict = predict+dif_predict
    predict_auto = predict_auto+dif_predict_auto
    print('time: ', time.time()-st)
    # predict_auto2 = predict_auto2 + dif_predict_auto2

    # plt.ylim(ymin=true[0]-2, ymax=true[0]+2)
    # plt.ylim(ymin=0, ymax=1)
    plt.plot(x, true, label='actual', color='green')
    # plt.plot(x[sequence_length-1:], predict, label='NN(H=2)', color='red')
    plt.plot(x[sequence_length-1:], predict_auto, label='linear bias model correction on seq2seq', color='orange')
    # plt.plot(x[sequence_length:], predict_auto2, label='seq2seq(H=3)', color='black')

    plt.legend()
    plt.title(filename+' dim:' + str(dim))
    plt.savefig('examples1/'+filename+' dim:' + str(dim) + '.png')
    plt.close()
    # plt.show()

if __name__ == '__main__':
    sequence_length = 3
    dim_list = [0, 1, 2, 6, 7, 8, 12, 13, 14, 21, 28, 35]
    for dim in dim_list:
        plot_com('results_auencoder3_C', filename='10.5100.75', dim=dim)
        plot_com('results_auencoder3_C', filename='10.2500.750.25', dim=dim)
        plot_com('results_auencoder3_C', filename='0.250.250.7511', dim=dim)

    # added_linear('results_auencoder3_C', 'results_auencoder3_S')