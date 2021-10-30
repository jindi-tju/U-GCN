import scipy.sparse as sp
import torch
import numpy as np
import os

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(config):
    feature_list = []
    label_list = []
    f = open('./data/{}/out1_node_feature_label.txt'.format(config.data_set), 'r')
    for line in f.readlines():
        ele = line.strip().split('\t')
        if ele[0] == 'node_id':
            continue
        feature = ele[1]
        label = int(ele[2])
        feature = feature.strip().split(',')
        feature_list.append(feature)
        label_list.append(label)
    feature = np.array(feature_list, dtype=float)

    idx = np.load(os.path.join("./data/{}/".format(config.data_set), 'train_val_test_idx.npz'))
    idx_train = idx['train_idx']
    idx_test = idx['test_idx']
    idx_val = idx['val_idx']

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(label_list)
    features = sp.csr_matrix(feature, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    return features, labels, idx_train, idx_val, idx_test

def load_graph(dataset, config):
    print('Loading {} dataset...'.format(dataset))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    adj_2_edges = np.genfromtxt(config.adjgraph_path, dtype=np.int32)
    adj2edges = np.array(list(adj_2_edges), dtype=np.int32).reshape(adj_2_edges.shape)
    adj2 = sp.coo_matrix((np.ones(adj2edges.shape[0]), (adj2edges[:, 0], adj2edges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - sadj.multiply(adj2.T > adj2)
    nsadj2 = normalize(adj2 + sp.eye(adj2.shape[0]))

    nsadj = torch.FloatTensor(np.array(nsadj.todense()))
    nsadj2 = torch.FloatTensor(np.array(nsadj2.todense()))
    nfadj = torch.FloatTensor(np.array(nfadj.todense()))

    return nsadj, nsadj2, nfadj

