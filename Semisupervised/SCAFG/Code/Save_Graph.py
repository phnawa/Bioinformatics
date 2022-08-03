import networkx as nx
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
import dgl
import torch
import os


Dataset = ['Chu_cell_type', 'Patel', 'Xin_human_islets', 'Chung', 'Ning']
Threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def Save_Graph(dataset):
    for k in trange(len(Threshold)):
        os.chdir('D:\Graduation_paper\Dataset\data')
        data_graph = joblib.load(dataset + '_threshold=%0.1f.pkl'%(Threshold[k]))

        data_graph = pd.DataFrame(data_graph)
        print(data_graph)
        g = nx.Graph()


        for i in trange(data_graph.shape[1]):
            for j in range(i, data_graph.shape[1]):
                if data_graph.T[i][j] != 0:
                    g.add_edge(i, j)


        ## DGL存图
        src, dst = tuple(zip(*g.edges()))
        G = dgl.DGLGraph()
        G.add_nodes(data_graph.shape[0])
        G.add_edges(src, dst)
        cuda_G = G.to('cuda:0')
        # print(cuda_G.device)
        # edges are directional in DGL; make them bi-directional


        print(cuda_G)
        print(cuda_G.edges())
        print('We have %d nodes.' % cuda_G.number_of_nodes())
        print('We have %d edges.' % cuda_G.number_of_edges())
        cuda_G.ndata['feat'] = torch.eye(data_graph.shape[0]).to('cuda:0')
        # joblib.dump(g, dataset + 'Graph_Info_' + str(Threshold[k]) + '.pkl')
        joblib.dump(cuda_G, dataset + 'Graph_Info_' + str(Threshold[k]) + '.pkl')
    # print(g[0][1])
    # print(g.edges())
    # print(g._node)


def Save_Consensus_Graph(dataset):
    os.chdir('D:\Graduation_paper\Dataset\data')
    data_graph = joblib.load(dataset + '_data_dict_aggregate_k=4.pkl')
    data_graph = pd.DataFrame(data_graph)
    print(data_graph)
    g = nx.Graph()


    for i in trange(data_graph.shape[0]):
        for j in range(i, data_graph.shape[1]):
            if data_graph.T[i][j] != 0:
                g.add_edge(i, j)


    ## DGL存图
    src, dst = tuple(zip(*g.edges()))
    G = dgl.DGLGraph()
    G.add_nodes(data_graph.shape[0])
    G.add_edges(src, dst)
    cuda_G = G.to('cuda:0')
    # print(cuda_G.device)
    # edges are directional in DGL; make them bi-directional


    print(cuda_G)
    print(cuda_G.edges())
    print('We have %d nodes.' % cuda_G.number_of_nodes())
    print('We have %d edges.' % cuda_G.number_of_edges())
    cuda_G.ndata['feat'] = torch.eye(data_graph.shape[0]).to('cuda:0')
    # joblib.dump(g, dataset + 'Graph_Info_' + str(Threshold[k]) + '.pkl')
    joblib.dump(cuda_G, dataset + '_Data_Dict_Consensus_Graph_Info_k=4.pkl')

for t in range(len(Dataset)):
    # Save_Graph(Dataset[t])
    Save_Consensus_Graph(Dataset[t])




