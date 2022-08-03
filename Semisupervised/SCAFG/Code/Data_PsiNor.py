import pandas as pd
import numpy as np
from tqdm import trange
import joblib
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import os
from tqdm import trange

Dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets',  'Chung', 'Ning']
title = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
# data = joblib.load('Chu_cell_type' + '.pkl')
# print(data.shape)

def data_diff(dataset):
    os.chdir('D:\Graduation_paper\Dataset\data')
    data = joblib.load(dataset + '.pkl')
    os.chdir('D:\Graduation_paper\Dataset\label')
    label = joblib.load(dataset + '_labels.pkl')
    data_PsiNor = np.zeros((data.shape[0], data.shape[1]))

    ## ***********  PisNorm  ***************
    for i in range(data.shape[0]):
        sum = 0
        for j in range(data.shape[1]):
            data_PsiNor[i,j] = abs(float(data[i,j]))
            sum += math.log((data_PsiNor[i,j])+1, 2)
        data_PsiNor[i,:] = data_PsiNor[i,:]*data_PsiNor.shape[1]/sum

    print(data, data.shape)
    # print(np.max(data, axis=0))
    print(data_PsiNor, data_PsiNor.shape)

    data_similarity = cosine_similarity(data_PsiNor)
    print(data_similarity, data_similarity.shape)
    # data_similarity = MinMaxScaler().fit_transform(data_similarity)

    os.chdir('D:\Graduation_paper\Dataset\data')
    joblib.dump(data_PsiNor, dataset + '_psiNorm_matrix.pkl')
    joblib.dump(data_similarity, dataset + '_psiNorm_similarity_matrix.pkl')

    # joblib.dump(data_mahanttan, dataset + '_manhattan_matrix.pkl')
    # sns.heatmap(data_similarity, cbar=True)
    # print(data_similarity, data_similarity.shape)
    # print(data_mahanttan, data_mahanttan.shape)

    return data_PsiNor, data_similarity



def draw_hist(dataset):
    data = joblib.load(dataset + '_similarity_matrix.pkl')
    data_cnt = [ 0 for i in range(9)]
    threshold = [0.1*(i+1) for i in range(9)]
    for i in trange(data.shape[0]):
        for j in range(i, data.shape[1]):
            for s in range(len(threshold)):
                if data[i,j] > threshold[s]:
                    data_cnt[s] += 1

    plt.title(dataset)
    plt.plot(threshold, data_cnt)
    plt.show()

for t in trange(len(Dataset)):
    # plt.subplot(2,2,t+1)
    # plt.title(title[t])
    data_diff(Dataset[t])
    # draw_hist(Dataset[t])

# plt.show()


