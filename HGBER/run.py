import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import time
import sys

from models import DGI, LogReg, Net_dg, H, clustering
from utils import process
import data_process
import misc
from clf_result import clf, clf_lr

from settings import set
settings = set()
inter, loss_inter, L = [], [], []

parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--self_traning', action='store_true', default=True, help='Using self training loss or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=settings['latent_dim'], help='Number of hidden units.')
parser.add_argument('--shid', type=int, default=8, help='Number of semantic level hidden units.')
parser.add_argument('--out', type=int, default=8, help='Number of output feature dimension.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=30, help='Patience')
args = parser.parse_args()
print('Running dataset: ' + settings['data_set'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', settings['GPU'])
print('Using GPU: '+str(settings['GPU']))

# training params
batch_size = 1
nb_epochs = args.epochs
patience = args.patience
lr = args.lr
l2_coef = args.weight_decay
drop_prob = args.dropout
hid_units = args.hidden
sparse = args.sparse
nonlinearity = 'prelu'  # special name to separate parameters

Data = data_process.loadgraph(settings)
settings['num_node'] = Data['features'].size(0)
settings['in_dim'] = Data['features'].size(1)
if settings['data_set'] == 'IMDB':
    settings['i'] = Data['i']

adjs, features, labels, idx_train, idx_val, idx_test = process.load_mat(settings['data_set'])

features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nor_adjs = []
for adj in adjs:
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = adj[np.newaxis]
    nor_adjs.append(adj)
nor_adjs = torch.FloatTensor(np.array(nor_adjs))

features = torch.FloatTensor(features[np.newaxis])

labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model_dgis = [DGI(ft_size, hid_units, nonlinearity).to(device) for i in range(settings['num_view'])]
model_dgs = [Net_dg(settings).to(device).to(device) for i in range(settings['num_view'])]
get_H = H(settings).to(device)
if args.self_traning:
    clustering = clustering(settings).to(device)

optimizer1 = torch.optim.Adam([{"params": dgi.parameters()} for dgi in model_dgis], lr=settings['lr_dgi'])
optimizer2 = torch.optim.Adam([{"params": model_dg.parameters()} for model_dg in model_dgs], lr=settings['lr_dg'])
optimizer3 = torch.optim.Adam(get_H.parameters(), lr=settings['lr_H'])
if args.self_traning:
    optimizer4 = torch.optim.Adam(clustering.parameters(), lr=settings['lr_cluster'])

features = features.to(device)
nor_adjs.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
mse = nn.MSELoss()
emb_dgi, z_dg = {}, {}
for dgi in model_dgis:
    dgi.train()
    for model_dg in model_dgs:
        model_dg.train()
idx = np.random.permutation(nb_nodes)
shuf_fts = features[:, idx, :]

lbl_1 = torch.ones(batch_size, nb_nodes)
lbl_2 = torch.zeros(batch_size, nb_nodes)
lbl = torch.cat((lbl_1, lbl_2), 1)

shuf_fts = shuf_fts.to(device)
lbl = lbl.to(device)
nor_adjs = nor_adjs.to(device)
emb_list = []
for epoch in range(settings['prtraindgi_EPOCH']):
    optimizer1.zero_grad()
    for num, dgi in enumerate(model_dgis):
        logits = dgi(features, shuf_fts, nor_adjs[num], sparse, None, None, None)
        emb_dgi[str(num)], _ = dgi.embed(features, nor_adjs[num], sparse, None)
        loss = b_xent(logits, lbl)
        loss.backward()
        # print(loss)
        # L.append(loss)
        if settings['data_set'] == 'IMDB':
            emb = emb_dgi[str(num)].detach().cpu().data.numpy()[settings['i']]
        else:
            emb = emb_dgi[str(num)].detach().cpu().data.numpy()
        if epoch > 10 and epoch % 5 == 0:
            emb_list.append(emb)
    if epoch > 50 and epoch % 5 == 0:
        a = np.concatenate((emb_list[0],emb_list[1]),axis=1)
        # b = np.concatenate((a,emb_list[2]),axis=1)
        c = a
        # c = np.concatenate((b,emb_list[3]),axis=1)
        # misc.visualizeData(b, Data['labels'].data.numpy(), settings, 'dgi_' + settings['data_set']+ str(epoch))
        emb_list.clear()
        acc, acc_sd, nmi, nmi_std, ari, ari_std = misc.evaluateKMeans(c, Data['labels'].data.numpy(), settings)
        print('dgi')
        print('epoch: %d' % epoch)
        print('prtrainH clusering : acc: %.4f--std: %.4f--nmi: %.4f--sd: %.4f--ari: %.4f--std: %.4f' % (acc, acc_sd, nmi, nmi_std, ari, ari_std))
        micro_f1, macro_f1 = clf_lr(c, Data['labels'].data.numpy(), 0.2)
        print('prtrainH classification : Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
    optimizer1.step()
# sys.exit()
for num, dgi in enumerate(model_dgis):
    logits = dgi(features, shuf_fts, nor_adjs[num], sparse, None, None, None)
    emb_dgi[str(num)], _ = dgi.embed(features, nor_adjs[num], sparse, None)
for epoch in range(settings['prtrainH_EPOCH']):
    H = get_H()
    optimizer2.zero_grad()
    for num, model_dg in enumerate(model_dgs):
        z = model_dg(H)
        loss_recon_dg = settings['lamb_dg'] * mse(z, emb_dgi[str(num)])
        # acc, nmi = misc.evaluateKMeans(Z_ae[str(view)].detach().cpu().data.numpy(), Data['labels'].data.numpy(), settings['num_class'])
        # print(str(view)+':')
        # print(acc)
        loss_recon_dg.backward()
    optimizer2.step()
    for i in range(settings['inter_H']):
        optimizer3.zero_grad()
        for num, model_dg in enumerate(model_dgs):
            z = model_dg(H)
            loss_recon_h = settings['view' + str(num)] * mse(z, emb_dgi[str(num)])
            loss_recon_h.backward()
            # print(loss_recon_h)
        optimizer3.step()
    H = get_H()
    for num, model_dg in enumerate(model_dgs):
        z_dg[str(num)] = model_dg(H)
    optimizer1.zero_grad()
    for num, dgi in enumerate(model_dgis):
        logits = dgi(features, shuf_fts, nor_adjs[num], sparse, None, None, None)
        emb_dgi[str(num)], _ = dgi.embed(features, nor_adjs[num], sparse, None)
        loss = b_xent(logits, lbl) + settings['view' + str(num)] * mse(z_dg[str(num)], emb_dgi[str(num)])
        loss.backward()
    optimizer1.step()
    # if settings['data_set'] == 'IMDB':
    #     emb = H.detach().cpu().data.numpy()[settings['i']]
    # else:
    #     emb = H.detach().cpu().data.numpy()
    # acc, acc_sd, nmi, nmi_std, ari, ari_std = misc.evaluateKMeans(emb,
    #                                                                   Data['labels'].data.numpy(), settings)
    # print('H')
    # print('epoch: %d' % epoch)
    # misc.visualizeData(emb, Data['labels'].data.numpy(), settings, 'h_' + settings['data_set']+ str(epoch))
    # print('prtrainH clusering : acc: %.4f--std: %.4f--nmi: %.4f--sd: %.4f--ari: %.4f--std: %.4f' % (
    #     acc, acc_sd, nmi, nmi_std, ari, ari_std))
    # micro_f1, macro_f1 = clf_lr(emb, Data['labels'].data.numpy(), 0.2)
    # print('prtrainH classification : Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
H = get_H()
kmeans = KMeans(n_clusters=settings['num_class'], n_init=20)
kmeans.fit(H.detach().cpu().data.numpy())
clustering.clustering_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
for epoch in range(nb_epochs):
    inter.append(epoch)
    H = get_H()
    for num, model_dg in enumerate(model_dgs):
        z_dg[str(num)] = model_dg(H)
    optimizer1.zero_grad()
    for num, dgi in enumerate(model_dgis):
        logits = dgi(features, shuf_fts, nor_adjs[num], sparse, None, None, None)
        emb_dgi[str(num)], _ = dgi.embed(features, nor_adjs[num], sparse, None)
        loss = b_xent(logits, lbl) + settings['view' + str(num)] * mse(z_dg[str(num)], emb_dgi[str(num)])
        loss.backward()
    optimizer1.step()
    optimizer2.zero_grad()
    for num, model_dg in enumerate(model_dgs):
        z = model_dg(H)
        loss_recon_dg = settings['lamb_dg'] * mse(z, emb_dgi[str(num)])
        # acc, nmi = misc.evaluateKMeans(Z_ae[str(view)].detach().cpu().data.numpy(), Data['labels'].data.numpy(), settings['num_class'])
        # print(str(view)+':')
        # print(acc)
        loss_recon_dg.backward()
    optimizer2.step()
    for i in range(settings['inter_H']):
        H = get_H()
        cluster = clustering()
        optimizer3.zero_grad()
        q, p = data_process.target_distribution(H, cluster)
        loss_klh = settings['lamb_cluster'] * F.kl_div(q.log(), p.clone().detach(), reduction='batchmean')
        loss_klh.backward()
        for num, model_dg in enumerate(model_dgs):
            z = model_dg(H)
            loss_recon_h = settings['view' + str(num)] * mse(z, emb_dgi[str(num)])
            loss_recon_h.backward()
        optimizer3.step()
    H = get_H()
    for i in range(settings['inter_cluster']):
        optimizer4.zero_grad()
        cluster = clustering()
        q, p = data_process.target_distribution(H, cluster)
        loss_kl = F.kl_div(q.log().clone().detach(), p, reduction='batchmean')
        loss_kl.backward()
        # print(loss_kl)
        optimizer4.step()
    H = get_H()
    cluster = clustering()
    q, p = data_process.target_distribution(H, cluster)
    for num in range(settings['num_view']):
        logits = model_dgis[num](features, shuf_fts, nor_adjs[num], sparse, None, None, None)
        em, _ = model_dgis[num].embed(features, nor_adjs[num], sparse, None)
        z = model_dgs[num](H)
        loss_k = 0.001*b_xent(logits, lbl) + settings['view' + str(num)] * mse(z, em)
        L.append(loss_k)
    loss_kl = F.kl_div(q.log().clone().detach(), p, reduction='batchmean')
    loss_total = sum(L) + loss_kl
    loss_inter.append(loss_total)
    L.clear()
    if epoch % 5 == 0:
        H = get_H()
        if settings['data_set'] == 'IMDB':
            emb = H.detach().cpu().data.numpy()[settings['i']]
        else:
            emb = H.detach().cpu().data.numpy()
        acc, acc_sd, nmi, nmi_std, ari, ari_std = misc.evaluateKMeans(emb,
                                                                      Data['labels'].data.numpy(), settings)
        print('H+self')
        print('epoch: %d' % epoch)
        # misc.visualizeData(emb, Data['labels'].data.numpy(), settings, 'hself_' + settings['data_set']+ str(epoch))
        print('prtrainH clusering : acc: %.4f--std: %.4f--nmi: %.4f--sd: %.4f--ari: %.4f--std: %.4f' % (
        acc, acc_sd, nmi, nmi_std, ari, ari_std))
        micro_f1, macro_f1 = clf_lr(emb, Data['labels'].data.numpy(), 0.2)
        print('prtrainH classification : Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro_f1, macro_f1))
        # misc.visualizeData(emb, Data['labels'].data.numpy(), settings,
        #                    'H_' + settings['data_set'] + str(epoch))

        # k = time.strftime('%m-%d-%H-%M', time.localtime(time.time())) + str(epoch) +'.png'
        # fig = plt.figure(figsize=(10, 6))
        # plt.rcParams['figure.dpi'] = 200
        
        # plt.tick_params(labelsize=15)
        # plt.plot(inter, loss_inter, linestyle='--', linewidth=4, color='green', markersize=6, markerfacecolor='brown')
        # plt.xlabel('Interation', fontsize=20)
        # plt.ylabel('Value of objective function', fontsize=20)
        # plt.savefig(k)

