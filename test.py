"""
Testing of IRCNN
@author: LQ
"""
import pickle
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn import metrics
import module
from module import IRCNN
import yaml


class Tester(object):
    def __init__(self, cfig):
        self.cfig = cfig
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IRCNN(in_channels=6, out_channels=64,kernel_size=3, num_classes=2,time_len=10).to(self.device)

    def test(self):

        print('Testing..................')

        model_root = osp.join(self.cfig['save_path'], 'models')
        best_model_pth = '%s/best_model_last.pth' % (model_root)
        self.model.load_state_dict(torch.load(best_model_pth))
        path_test = self.cfig['data_path']

        with open(os.path.join(path_test, 'test_sample.pickle'), 'rb') as file:
            test_sample, test_label, test_dist = pickle.load(file)

        test_sz = test_label.shape[0]
        BATCH_SZ = test_sz
        iter_in_test = test_sz // BATCH_SZ
        test_idx = np.arange(0, test_sz)
        pred = []
        real = []

        for _iter in range(iter_in_test):

            batch_test, batch_label, batch_dist = self.get_batch_data(_iter, BATCH_SZ, test_idx, test_sample,test_label, test_dist)
            batch_test, batch_label, batch_dist = batch_test.to(self.device), batch_label.to(self.device), batch_dist.to(self.device)

            pred_list, target_list = self.test_epoch(batch_test, batch_label, batch_dist)

            pred.extend(pred_list)
            real.extend(target_list)


        OA = accuracy_score(real, pred)
        precision = precision_score(real, pred, pos_label=1)
        recall = recall_score(real, pred, pos_label=1)
        f1 = f1_score(real, pred, pos_label=1)
        fpr, tpr, threshold = metrics.roc_curve(real, pred)
        roc_auc = metrics.auc(fpr, tpr)
        Kappa = self.kappa(confusion_matrix(real, pred))
        print(confusion_matrix(real, pred))
        print('OA: %.4f  Kappa: %.4f     F1: %.4f' %(OA, Kappa, f1))

    def test_epoch(self, batch_train, batch_label, batch_dist):
        self.model.eval()
        pred_list, target_list = [], []
        self.optim.zero_grad()
        pred = self.model(batch_train, batch_dist)
        pred_prob = torch.sigmoid(pred)
        pred_cls = pred_prob.data.max(1)[1]
        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        return pred_list, target_list

    def get_batch_data(self, _iter, BATCH_SZ, idx, sample, label, dist):
        start_idx = _iter * BATCH_SZ
        end_idx = (_iter + 1) * BATCH_SZ
        batch_sample = sample[:, idx[start_idx:end_idx], :, :, :]
        batch_label = label[idx[start_idx:end_idx]]
        batch_dist = dist[idx[start_idx:end_idx]]
        return batch_sample, batch_label, batch_dist

    def kappa(self, confusion_matrix):
        pe_rows = np.sum(confusion_matrix, axis=0)
        pe_cols = np.sum(confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

if __name__ == '__main__':
    f = open('Parameter.yaml', 'r').read()
    cfig = yaml.load(f)
    tester = Tester(cfig)
    tester.test()


