from itertools import permutations
import os
import joblib
import numpy as np
from itertools import combinations


def distance(a, b):
    cnt = 0
    for i in range(a.shape[0]):
        if a[i] == b[i]:
            cnt += 1
    return cnt


dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
data_dict = {}
data_merge_dict ={}


for k in range(len(dataset)):
    dis_list = np.array([[0 for p in range(9)] for q in range(9)])
    os.chdir('D:\Graduation_paper\Dataset\label')
    label = joblib.load(dataset[k] + '_labels.pkl')
    print(dataset[k]+'\t')
    for t in range(len(threshold)-1):
        os.chdir('D:\Graduation_paper\Dataset\data')
        label_pred1 = joblib.load(
            dataset[k]+'_Pred_label(epoch=75, threhold=%0.1f, init_node=%d).pkl' % ((threshold[t]), int(label.shape[0]//10)))
        for s in range(t+1, len(threshold)):
            label_pred2 = joblib.load(
                dataset[k]+'_Pred_label(epoch=75, threhold=%0.1f, init_node=%d).pkl' % ((threshold[s]), int(label.shape[0]//10)))
            dis_list[t][s] = distance(label_pred1, label_pred2)
            dis_list[s][t] = dis_list[t][s]
    print(dis_list, dis_list.shape)

    tmp = dis_list
    loc = np.argmax(dis_list)
    r, c = divmod(loc, dis_list.shape[1])
    print(r, c)
    data_dict.setdefault(dataset[k], []).append(r)
    data_dict.setdefault(dataset[k], []).append(c)
    # print(data_dict)

    tmp[r][c] = 0
    tmp[c][r] = 0
    loc = np.argmax(tmp[:,c])
    m, n = divmod(loc, len(tmp[:,c]))
    print(m, n)
    data_dict.setdefault(dataset[k], []).append(n)
    # print(data_dict)

    tmp[c][n] = 0
    tmp[n][c] = 0
    loc = np.argmax(tmp[r])
    m, n = divmod(loc, len(tmp[r]))
    print(m, n)
    data_dict.setdefault(dataset[k], []).append(n)


    joblib.dump(dis_list, dataset[k]+'_dislist.pkl')

print(data_dict)
joblib.dump(data_dict, 'Data_dict.pkl')


for i in range(len(dataset)):
    data_merge_dict[dataset[i]] = data_dict[dataset[i]]
    data_merge_dict[dataset[i]] = list(set(data_merge_dict[dataset[i]]))


data_merge_dict.setdefault('Xin_human_islets', []).append(5) 
data_merge_dict.setdefault('Chung', []).append(3)
data_merge_dict.setdefault('Ning', []).append(6)


print(data_merge_dict)
joblib.dump(data_merge_dict, 'Data_Merge_Dict.pkl')


# for x in range(len(dataset)):
#     dis_matrix = joblib.load(dataset[x]+'_dislist.pkl')
#     # tmp存放每次dis_matrix的加和，sv存放tmp与上次tmp的较大值，cnt存放sv对应的索引m。
#     sv, cnt = 0, 0
#     for m in combinations(threshold, 5):
#         tmp = 0
#         for n in range(len(m)-1):
#             for y in range(n+1, len(m)):
#                 tmp += dis_matrix[int(10*m[n]-1)][int(10*m[y]-1)]
#         if tmp >= sv:
#             sv = tmp
#             cnt = m
#     print(dataset[x]+'\t')
#     print(sv, cnt)
#     cnt = list(cnt)
#     cnt.append(sv)
#     for z in range(len(cnt)):
#         data_dict.setdefault(dataset[x], []).append(cnt[z])


# print('data_dict_k=5'+'\t')
# print(data_dict)
# joblib.dump(data_dict, 'Data_dict_k=5.pkl')
