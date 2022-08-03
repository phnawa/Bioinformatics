import joblib
import numpy as np
from tqdm import trange
import os


Dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
Threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def threshold_division(dataset):
    os.chdir('D:\Graduation_paper\Dataset\data')
    for i in trange(len(dataset)):
        data = joblib.load(dataset[i] + '_psiNorm_similarity_matrix.pkl')
        for j in trange(len(Threshold)):
            data_threshold = np.zeros((data.shape[0], data.shape[1]))
            for k in trange(data.shape[0]):
                for t in range(data.shape[1]):
                    if data[k][t] >= Threshold[j]:
                        data_threshold[k][t] = 1
                    else:
                        data_threshold[k][t] = 0
            joblib.dump(data_threshold, dataset[i] + '_threshold=' + str(Threshold[j]) + '.pkl')

threshold_division(Dataset)
