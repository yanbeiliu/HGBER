from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix


import torch

def loadgraph(settings):
    DATA = {}
    if settings['data_set'] == 'ACM':
        data = sio.loadmat(settings['data_dir']+'ACM3025.mat')
        data['label'] = [list(one_label).index(1) for one_label in list(data['label'])]
        DATA['labels'], DATA['features'] = torch.tensor(data['label'], dtype=torch.long), torch.tensor(norm(data['feature']), dtype=torch.float)
        DATA['train_index'], DATA['val_index'], DATA['test_index'] = list(data['train_idx']), list(data['val_idx']), list(data['test_idx'])

        DATA['edge1'], DATA['edge0'] = torch.tensor([coo_matrix(data['PAP']).row, coo_matrix(data['PAP']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['PLP']).row, coo_matrix(data['PLP']).col], dtype=torch.long)
        return DATA
    if settings['data_set'] == 'DBLP':
        data = sio.loadmat(settings['data_dir'] + 'DBLP4057_GAT_with_idx.mat')
        data['label'] = [list(one_label).index(1) for one_label in list(data['label'])]
        DATA['labels'], DATA['features'] = torch.tensor(data['label'], dtype=torch.long), torch.tensor(norm(data['features']), dtype=torch.float)
        DATA['train_index'], DATA['val_index'], DATA['test_index'] = list(data['train_idx']), list(data['val_idx']), list(data['test_idx'])
        DATA['edge0'], DATA['edge1'], DATA['edge2'] = torch.tensor([coo_matrix(data['net_APA']).row, coo_matrix(data['net_APA']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['net_APCPA']).row, coo_matrix(data['net_APCPA']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['net_APTPA']).row, coo_matrix(data['net_APTPA']).col], dtype=torch.long)
        return DATA
    if settings['data_set'] == 'IMDB':
        data = sio.loadmat(settings['data_dir'] + 'imdb.mat')
        DATA['i'] = data['i_label'][0]-1
        data['label'] = [list(one_label).index(1) for one_label in list(data['label'][DATA['i']])]
        DATA['labels'], DATA['features'] = torch.tensor(data['label'], dtype=torch.long), torch.tensor(data['feature'], dtype=torch.float)
        DATA['train_index'], DATA['val_index'], DATA['test_index'] = list(data['train_idx']), list(data['val_idx']), list(data['test_idx'])
        DATA['edge0'], DATA['edge1'] = torch.tensor([coo_matrix(data['MAM']).row, coo_matrix(data['MAM']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['MDM']).row, coo_matrix(data['MDM']).col], dtype=torch.long)
        return DATA
    if settings['data_set'] == 'Yelp':
        data = sio.loadmat(settings['data_dir'] + 'yelp2614.mat')
        data['label'] = [list(one_label).index(1) for one_label in list(data['label'])]
        DATA['labels'], DATA['features'] = torch.tensor(data['label'], dtype=torch.long), torch.tensor(norm(data['features']), dtype=torch.float)
        DATA['edge0'], DATA['edge1'], DATA['edge2'], DATA['edge3'] = torch.tensor([coo_matrix(data['BUB']).row, coo_matrix(data['BUB']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['BTB']).row, coo_matrix(data['BTB']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['BSB']).row, coo_matrix(data['BSB']).col], dtype=torch.long), \
                                                            torch.tensor([coo_matrix(data['BRB']).row, coo_matrix(data['BRB']).col], dtype=torch.long)
        return DATA
def bulid_graph(num_view, DATA, device):
    G = {}
    for view in range(num_view):
        G[str(view)] = Data(x=DATA['features'], y=DATA['labels'], edge_index=DATA['edge'+str(view)]).to(device)
    return G

def norm(features):
    s = MinMaxScaler()
    return s.fit_transform(features)
    
def target_distribution(H, cluster, v=1):
    q = 1.0 / (1.0 + torch.sum(torch.pow(H.unsqueeze(1) - cluster, 2), 2) / v)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return q, p
    


