import joblib
import numpy as np
from tqdm import trange
import os


dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
os.chdir('D:\Graduation_paper\Dataset\data')
data_dict = joblib.load('Data_Merge_Dict.pkl')


def DataDict_Aggregate(dataset, k):
    data_aggregate_list = []
    data_aggreate = []
    for x in range(k):
        data_aggregate_list.append(dataset + '_threshold=' + str((data_dict[dataset][x]+1)/10))
        data_aggregate_list[x] = joblib.load(dataset + '_threshold=' + str((data_dict[dataset][x]+1)/10) + '.pkl')
    for i in trange(data_aggregate_list[k-1].shape[0]):
        tmp = []
        for j in range(data_aggregate_list[k-1].shape[1]):
            cnt1, cnt2 = 0, 0
            for s in range(k):
                if data_aggregate_list[s][i][j] == 0:
                    cnt1 += 1
                if data_aggregate_list[s][i][j] == 1:
                    cnt2 += 1
            if(cnt1 > cnt2):
                tmp.append(0)
            else:
                tmp.append(1)
        data_aggreate.append(tmp)
    data_aggreate = np.array(data_aggreate)
    print(data_aggreate)
    joblib.dump(data_aggreate, dataset + '_data_dict_aggregate_k=%d.pkl'%(k))


for x in range(len(dataset)):
    DataDict_Aggregate(dataset[x], 4)


# def consensus_matrix(dataset, k):
#     consensus_matrix = []
#     data_threshold_list = []
#     os.chdir('D:\Graduation_paper\Dataset\data')
#     for t in range(k):
#         data_threshold_list.append(dataset + '_threshold=' + str(Threshold[t]))
#         data_threshold_list[t] = joblib.load(dataset + '_threshold=' + str(Threshold[t]) + '.pkl')
    # for i in trange(data_threshold_list[k-1].shape[0]):
    #     tmp = []
    #     for j in range(data_threshold_list[k-1].shape[1]):
    #         cnt1, cnt2 = 0, 0
    #         for s in range(k):
    #             if data_threshold_list[s][i][j] == 0:
    #                 cnt1 += 1
    #             if data_threshold_list[s][i][j] == 1:
    #                 cnt2 += 1
    #         if(cnt1 > cnt2):
    #             tmp.append(0)
    #         else:
    #             tmp.append(1)
    #     consensus_matrix.append(tmp)
    # consensus_matrix = np.array(consensus_matrix)
#     joblib.dump(consensus_matrix, dataset + '_auto_consensus_matrix.pkl')






