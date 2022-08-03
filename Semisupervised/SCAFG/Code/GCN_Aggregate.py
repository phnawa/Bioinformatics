from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import Counter
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import joblib
import networkx as nx
from sklearn.decomposition import PCA
from tqdm import trange
import numpy as np
import os

def gcn_message(edges):

    # 参数：batch of edges
    # 得到计算后的batch of edges的信息，这里直接返回边的源节点的feature.
    return {'msg' : edges.src['h']}


def gcn_reduce(nodes):
    # 参数：batch of nodes.
    # 得到计算后batch of nodes的信息，这里返回每个节点mailbox里的msg的和
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}



# 定义GCN模型
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)


    def forward(self, g, inputs):
        # g 为图对象； inputs 为节点特征矩阵
        # 设置图的节点特征
        g.ndata['h'] = inputs
        # 触发边的信息传递
        # g.send(g.edges(), gcn_message)
        # # 触发节点的聚合函数
        # g.recv(g.nodes(), gcn_reduce)
        g.send_and_recv(g.edges(), gcn_message, gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        # 线性变换
        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


# '#00FF00', '#9400D3', '#7B68EE', '#808080', '#00FF00', '#808080', '#00BFFF'
# def draw(i):
#     # color = ['#FF0000', '#FF1493','#00FF00', '#9400D3',
#     #          '#7B68EE', '#808080', '#9400D3', '#FFB6C1']
#     pos = [[] for i in range(label.shape[0])]
#     # colors = []
#     for v in range(label.shape[0]):
#         pos[v] = all_logits[i][v].cpu().numpy()
#         cls = pos[v].argmax()
#         # colors.append(color[cls])
#     ax.cla()
#     ax.axis('off')
#     ax.set_title('Epoch: %d' % i)


#     pca = PCA(n_components=2)
#     sv = []
#     for j in range(label.shape[0]):
#         sv.append(pos[j])
#     sv = pca.fit_transform(sv)
#     os.chdir('D:\Graduation_paper\Dataset\data')
#     joblib.dump(sv, dataset + '_threshold=%.1f_dimension_reduction.pkl'%(threshold[k]))
#     for j in range(label.shape[0]):
#         pos[j] = sv[j]
    # nx.draw_networkx(nx_G, pos, style='--', width=0.5, node_color=colors,
    #                  edge_color='pink', node_size=50, font_size=3, font_color='black')


# 主要定义message方法和reduce方法
dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Max_Acc_Aggregate_dict= {}

for x in range(len(dataset)):
    Acc_threshold = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir('D:\Graduation_paper\Dataset\label')
    label = joblib.load(dataset[x] + '_labels.pkl')
    random_node = torch.tensor(np.random.randint(0, label.shape[0], size=(1, int(label.shape[0]//10))))
    random_label = torch.tensor([label[i] for i in random_node[0]])


    os.chdir('D:\Graduation_paper\Dataset\data')
    DGLGraph_Infos = joblib.load(dataset[x] + '_Data_Dict_Consensus_Graph_Info_k=2.pkl')
    print(DGLGraph_Infos.to(device))
    nx_G = DGLGraph_Infos.to_networkx().to_undirected()


    # colors = ['r', 'b', 'g', 'm']
    epochs = [25, 50, 75]
    Acc = [[] for t in range(len(epochs))]
    markers = ['.', '1', '2', '3']
    # plt.figure(figsize=(15, 7))
    # plt.title('Train_Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    for i in trange(len(epochs)):
        train_loss = []
        # plt.subplot(2,2,i+1)
        net = GCN(label.shape[0], 256, len(Counter(label).keys())).to('cuda:0')
        # net = GCN(672, 256, 6)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        all_logits = []
        inputs = torch.eye(label.shape[0]).to('cuda:0')
        ## 已知的节点和对应类别
        labeled_nodes = random_node[0].type(torch.long).to('cuda:0')
        labels = random_label.to('cuda:0')


        for epoch in trange(epochs[i]):
            acc, MaxAcc= 0, 0
            logits = net(DGLGraph_Infos.to(device), inputs).to(device)
            # we save the logits for visualization later
            all_logits.append(logits.detach())
            logp = F.log_softmax(logits, 1)
            idx = np.argmax(logp.cpu().detach().numpy(), axis=1)
            for j in range(len(label)):
                if label[j] == idx[j]:
                    acc += 1
            acc = acc/len(label)
            MaxAcc = max(MaxAcc, acc)
            Acc[i].append(acc)
            # we only compute loss for labeled nodes
            # 取label_nodes的index的logp行
            label_pred = logp[labeled_nodes]
            loss = F.nll_loss(label_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch %d | Loss: %.4f | Acc: %.4f' % (epoch, loss.item(), acc))
            train_loss.append(loss.item())
            if epoch == epochs[i]-1 :
                Max_Acc_Aggregate_dict.setdefault(dataset[x], []).append(MaxAcc)


        # 静态图
        # fig = plt.figure(dpi=150)
        # fig.clf()
        # ax = fig.subplots()
        # draw(epochs[i]-1)
        # os.chdir('E:\Program Files\python files\Graduation_paper\Figure')
        # plt.savefig(dataset + '_Graph_Train_Dimension_Reduction(init_node=%d, iteration=%d).png'%(len(random_node[0]), epochs[i]), bbox_inches='tight')




        ## 损失函数图像
        # plt.plot(train_loss, marker = markers[i], label = '(Epoch=%d)'%(epochs[i]))
        # plt.title('Train Loss(epoch=%d).svg' % (epochs[i]))
        # plt.grid(True)
        # plt.legend()


        # ## 动态图
        # fig = plt.figure(dpi=150)
        # fig.clf()
        # ax = fig.subplots()
        # draw(0)  # draw the prediction of the first epoch
        # ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
        # # plt.show()
        # ani.save('Dynamic_Graph_Train(epoch=%d).gif' % (epochs[i]), writer='pillow')


    Acc_threshold.append(Acc)
    os.chdir('D:\Graduation_paper\Figure')
    # plt.savefig(dataset[x] + '_Aggregate_Train_Loss(init_node=%d).svg'%(len(random_node[0])), bbox_inches='tight')
    # plt.close()
    # plt.show()
    os.chdir('D:\Graduation_paper\Dataset\data')
    joblib.dump(Acc_threshold, dataset[x] + '_Data_Dict_Aggregate_Acc_threshold_list(init_node=%d)_k=2.pkl'%(len(random_node[0])))

joblib.dump(Max_Acc_Aggregate_dict, 'Max_Aggreagte_Acc_Rate_k=2.pkl')


