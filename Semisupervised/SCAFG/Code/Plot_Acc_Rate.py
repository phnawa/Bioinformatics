from pickle import TRUE
import joblib
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange
import torch
import numpy as np
import matplotlib.cm as cm


Dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
# threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# for k in range(len(Dataset)):
#     os.chdir('D:\Graduation_paper\Dataset\label')
#     label = joblib.load(Dataset[k] + '_labels.pkl')
#     os.chdir('D:\Graduation_paper\Dataset\data')
#     plot_data = joblib.load(Dataset[k] + '_Acc_threshold_list(init_node=%d).pkl' % (int(label.shape[0]//10)))
#     random_node = torch.tensor(np.random.randint( 0, label.shape[0], size=(1, int(label.shape[0]//10))))
#     plt.rcParams['axes.unicode_minus'] = True
#     plt.rcParams['font.sans-serif'] = 'SimHei'
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     plt.rcParams['axes.titlesize'] = 10
#     plt.rcParams['xtick.direction'] = 'in'
#     plt.rcParams['ytick.direction'] = 'in'
#     plt.figure(figsize=(10, 10))

#     for i in trange(len(threshold)):
#         plt.subplot(3, 3, i+1)
#         epochs = [25, 50, 75]
#         colors = ['red', 'blue', '#27ae60']
#         if(i == 6 or i == 7 or i == 8):
#             plt.xlabel('Epoch')
#         plt.xlim(20, 40, 60, 80)
#         plt.ylim(0, 1)
#         if (i == 0 or i == 3 or i == 6):
#             plt.ylabel('Acc')
#         # markers = ['.', '1', '2', '3']
#         plt.title('Acc_Rate(threshold=%.1f)' % (threshold[i]))

#         for j in range(len(epochs)):
#             plt.plot(plot_data[i][j], color=colors[j],
#                      label='(Epoch=%d)' % (epochs[j]), linewidth=1)
#             # plt.title('Train Loss(epoch=%d).svg' % (epochs[i]))
#             plt.grid(False)
#     #     os.chdir('E:\Program Files\python files\Graduation_paper\Figure')
#     #     plt.savefig(Dataset + '_Acc_Rate(threshold=%.1f, init_node=%d).svg' % (threshold[i],len(random_node[0])), bbox_inches='tight')
#     # plt.show()
#     plt.suptitle(Dataset[k], fontsize=15, x=0.5, y=0.93)
#     os.chdir('D:\Graduation_paper\Figure')
#     plt.legend(bbox_to_anchor=(1.05, 3.4), loc=2,
#                borderaxespad=0, fontsize='large')
#     plt.savefig(Dataset[k] + 'Subplot_Acc_Rate(init_node=%d).svg' %
#                 (len(random_node[0])), bbox_inches='tight')
#     plt.show()

# for k in range(len(Dataset)):
#     os.chdir('D:\Graduation_paper\Dataset\label')
#     label = joblib.load(Dataset[k] + '_labels.pkl')
#     os.chdir('D:\Graduation_paper\Dataset\data')
#     plot_data = joblib.load(Dataset[k] + '_Data_Dict_Aggregate_Acc_threshold_list(init_node=%d).pkl'%(int(label.shape[0]//10)))
#     epochs = [25, 50, 75]
#     colors = ['r', 'b', 'g', 'm']
#     plt.xlabel('Epoch')
#     plt.xlim(0, 75)
#     plt.ylim(0, 1)
#     plt.ylabel('Acc')
#     # markers = ['.', '1', '2', '3']
#     plt.title(Dataset[k]+ '_Aggregate_Acc_Rate' )

#     for j in range(len(epochs)):
#         plt.plot(plot_data[0][j], color=colors[j], label='(Epoch=%d)' % (epochs[j]), linewidth=2)
#         # plt.title('Train Loss(epoch=%d).svg' % (epochs[i]))
#         plt.grid(True)
#         plt.legend(loc = 0)
#     #     os.chdir('E:\Program Files\python files\Graduation_paper\Figure')
#     #     plt.savefig(Dataset + '_Acc_Rate(threshold=%.1f, init_node=%d).svg' % (threshold[i],len(random_node[0])), bbox_inches='tight')
#     # plt.show()
#     os.chdir('D:\Graduation_paper\Figure')
#     plt.savefig(Dataset[k] + '_Data_Aggregate_Acc_Rate(init_node=%d).svg' % (int(label.shape[0]//10)), bbox_inches='tight')
#     plt.show()


os.chdir('D:\Graduation_paper\Dataset\data')
data1 = joblib.load('Max_Aggreagte_Acc_Rate_k=2.pkl')
data = {'Chu_cell_type': [0.63, 0.92, 0.97], 'Patel': [0.78, 0.91, 0.96], 'Xin_human_islets': [
    0.88, 0.89, 0.91], 'UsoSkin': [0.30, 0.32, 0.35], 'Camp15': [0.28, 0.29, 0.27], 'Chung': [0.45, 0.65, 0.83], 'Ning': [0.90, 0.94, 0.97]}
data1 = data
data2 = joblib.load('Max_Aggreagte_Acc_Rate_k=3.pkl')
data3 = joblib.load('Max_Aggreagte_Acc_Rate_k=4.pkl')
print(data1)
print(data2)
print(data3)

for i in range(len(Dataset)):
    os.chdir('D:\Graduation_paper\Dataset\data')
    x = np.arange(25, 100, 25)
    data4 = joblib.load(Dataset[i] + '_Max_Acc_List.pkl')
    data5 = np.average(data4, axis=0)
    # print(data4)
    print(data5)
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    l1 = plt.plot(x, data5, marker='o', label='Average',
                  linewidth=4, markersize=15)
    l2 = plt.plot(x, data1[Dataset[i]], marker='v',
                  label='k=2', linewidth=4, markersize=15)
    l3 = plt.plot(x, data2[Dataset[i]], marker='+',
                  label='k=3', linewidth=4, markersize=15)
    l4 = plt.plot(x, data3[Dataset[i]], marker='*',
                  label='k=4', linewidth=4, markersize=15)
    plt.legend(prop={'size': 20})
    plt.show()
    # os.chdir('D:\Graduation_paper\Figure')
    # plt.savefig(Dataset[i] + 'Acc_Rate.svg')
