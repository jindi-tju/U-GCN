from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import SFGCN
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
###################

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    parse.add_argument('--nb_heads', type=int, default=8, help='number of head attentions')
    parse.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
    parse.add_argument('--patience', type=int, default=20, help='Patience value')

    from datetime import datetime
    a = datetime.now()

    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    save_point = os.path.join('../checkpoint', args.dataset)
    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        # if cuda:
        #     torch.cuda.manual_seed(config.seed)

    sadj, sadj2, fadj = load_graph(args.dataset, config)
    features, labels, idx_train, idx_val, idx_test = load_data(config)

    model = SFGCN(nfeat=config.fdim,
                  nhid1=config.nhid1,
                  nhid2=config.nhid2,
                  nclass=config.class_num,
                  dropout=config.dropout,
                  alpha=args.alpha,
                  nheads=args.nb_heads)

    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        sadj2 = sadj2.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
        idx_val = idx_val.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_acc = 0
    state_dict_early_model = None
    curr_step = 0
    patience = args.patience
    atten_list = []
    percent = 0.80
    for epoch in range(config.epochs):

        model.train()
        optimizer.zero_grad()
        output, att, emb1, emb2, emb3 = model(features, sadj, sadj2, fadj)
        train_loss = F.nll_loss(output[idx_train], labels[idx_train])

        train_acc = accuracy(output[idx_train], labels[idx_train])
        train_loss.backward()
        optimizer.step()

        # Validation for each epoch
        model.eval()
        with torch.no_grad():
            output, att, emb1, emb2, emb3 = model(features, sadj, sadj2, fadj)
            val_loss = F.nll_loss(output[idx_val], labels[idx_val])
            val_acc = accuracy(output[idx_val], labels[idx_val])

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            state_dict_early_model = model.state_dict()
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break
        #     state = {
        #         'model': model,
        #         'acc': best_acc,
        #         'epoch': epoch,
        #     }
        #
        # torch.save(state, os.path.join(save_point, '%s.t7' % (opt.model)))

    model.load_state_dict(state_dict_early_model)
    model.eval()
    with torch.no_grad():
        output, att, emb1, emb2, emb3 = model(features, sadj, sadj2, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')

    print("Test_acc" + ":" + str(acc_test))
    print("Macro_f1" + ":" + str(macro_f1))

    b=datetime.now()
    print("total time" + str((b-a).seconds))



    
    
