import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_percentage_error

def compare(true_array, pred_array):
    mape_list = []
    for i in range(true_array.shape[0]):
        true = true_array[i].reshape(1, -1)
        predict = pred_array[i].reshape(1, -1)
        true[np.where(true == 0)] = 0.00001
        mape = mean_absolute_percentage_error(true, predict, multioutput='raw_values').reshape(1, -1)
        # mape[np.where(true == 0)] = 0
        mape_list.append(mape[0])
    mape_list = np.array(mape_list)
    mape = np.average(mape_list, axis=0)
    return mape

def calculate_mape_matrices(base_path, ids):
    results = {}

    for folder_id in ids:
        folder_C = os.path.join(base_path, f'results_new{folder_id}_C')
        folder_S = os.path.join(base_path, f'results_new{folder_id}_S')

        mape_C_matrices = []
        mape_S_matrices = []

        for filename in os.listdir(folder_C):
            if filename.endswith('true_C.csv'):
                pred_filename = filename.replace('true_C.csv', 'predict_C.csv')

                true_C_path = os.path.join(folder_C, filename)
                pred_C_path = os.path.join(folder_C, pred_filename)

                true_S_path = true_C_path.replace('_C/', '_S/').replace('true_C.csv', 'true_S.csv')
                pred_S_path = pred_C_path.replace('_C/', '_S/').replace('predict_C.csv', 'predict_S.csv')

                true_C = np.loadtxt(true_C_path, delimiter=',')
                pred_C = np.loadtxt(pred_C_path, delimiter=',')

                true_S = np.loadtxt(true_S_path, delimiter=',')
                pred_S = np.loadtxt(pred_S_path, delimiter=',')

                # Extract only the last 6 steps for comparison
                future_steps = 6
                true_C = true_C[-future_steps:, :]
                pred_C = pred_C[-future_steps:, :]
                true_S = true_S[-future_steps:, :]
                pred_S = pred_S[-future_steps:, :]

                # Calculate MAPE matrix for each file
                mape_C_matrix = np.zeros((6, 6))
                mape_S_matrix = np.zeros((6, 6))

                for i in range(6):
                    mape_C_matrix[i, :] = mean_absolute_percentage_error(true_C[:, i*6:(i+1)*6], pred_C[:, i*6:(i+1)*6], multioutput='raw_values')
                    mape_S_matrix[i, :] = mean_absolute_percentage_error(true_S[:, i*6:(i+1)*6], pred_S[:, i*6:(i+1)*6], multioutput='raw_values')

                mape_C_matrices.append(mape_C_matrix)
                mape_S_matrices.append(mape_S_matrix)

        # Average MAPE matrices over all files in the folder
        avg_mape_C = np.mean(mape_C_matrices, axis=0)
        avg_mape_S = np.mean(mape_S_matrices, axis=0)

        results[folder_id] = {
            'mape_C_matrix': avg_mape_C,
            'mape_S_matrix': avg_mape_S
        }

        # Save matrices to files
        np.savetxt(f'mape_C_matrix_{folder_id}.csv', avg_mape_C, delimiter=',')
        np.savetxt(f'mape_S_matrix_{folder_id}.csv', avg_mape_S, delimiter=',')

    return results

def calculate_mape_for_odf(base_path, ids):
    with open("filenames.json", "r") as infile:
        filenames = json.load(infile)
    test_filenames = filenames['test_filenames']
    print(len(test_filenames))

    results = {}

    for folder_id in ids:
        folder_path = os.path.join(base_path, f'results_new{folder_id}')

        mape_values = []

        for filename in test_filenames:
            [t1, t2, t3, t4, t5] = filename

            truefilename = str(t1) + '_' + str(t2) + '_' + str(t3) + '_' + str(t4) + '_' + str(t5)
            true_odfs = np.loadtxt('ODF_new/' + truefilename + '.csv', delimiter=',').transpose()
            pred_filename = str(t1) + str(t2) + str(t3) + str(t4) + str(t5)
            pred_odfs = np.loadtxt(folder_path + '/' + pred_filename + '_predict.csv', delimiter=",")
            # print(true_odfs.shape, predict_odfs.shape)

            # Extract only the last 6 steps for comparison
            future_steps = 6
            pred_odfs = pred_odfs[-future_steps:, :]
            true_odfs = true_odfs[-future_steps:, :]

            # Calculate MAPE using the compare function
            mape = compare(true_odfs, pred_odfs)
            mape_values.append(mape)

        # Average MAPE for the folder
        avg_mape = np.mean(mape_values)
        results[folder_id] = avg_mape
    return results

if __name__ == '__main__':
    base_path = './'  # Replace with the actual path to the results folders
    ids = [2, 3, 4, 5]

    # results = calculate_mape_matrices(base_path, ids)
    #
    # for folder_id, metrics in results.items():
    #     print(f"Folder ID {folder_id}:")
    #     print(f"  MAPE matrix for C saved as 'mape_C_matrix_{folder_id}.csv'")
    #     print(f"  MAPE matrix for S saved as 'mape_S_matrix_{folder_id}.csv'")

    results = calculate_mape_for_odf(base_path, ids)

    for folder_id, avg_mape in results.items():
        print(f"Folder ID {folder_id}: Average MAPE = {avg_mape}")
