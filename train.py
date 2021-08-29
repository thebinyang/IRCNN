"""
Train and Valid of IRCNN
"""
import pickle
import os
import os.path as osp
import torch
import yaml
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from module import IRCNN
from loss import BCE_weight_LOSS



class Trainer(object):
    def __init__(self, cfig):
        self.cfig = cfig
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IRCNN(in_channels=6, out_channels=64,kernel_size=3, num_classes=2,time_len=10).to(self.device)

    def train(self):
        print('Training..................')
        path = self.cfig['data_path']
        train_sample, train_label, train_dist = self.load_train_data(self, path)
        val_sample, val_label, val_dist = self.load_valid_data(self, path)
        train_idx = np.arange(0, train_label.shape[0])
        val_idx = np.arange(0, val_label.shape[0])
        best_val_roc_auc = 0
        BATCH_SZ = self.cfig['batch_size']
        iter_train_epoch = train_label.size // BATCH_SZ
        iter_val_epoch = val_label.size // BATCH_SZ

        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            val_ave_accuracy = 0
            
            # adjust learning rate in [10,15,20,25] epoch
            self.adjust_learning_rate(self.optim, epoch, [10,15,20,25], 0.5) 
            self.optim = torch.optim.Adam(self.params, betas=(0.9, 0.999))

            # trianing in every epoch
            for _iter in range(iter_train_epoch):  
                batch_train, batch_label, batch_dist = self.get_batch_data(_iter, BATCH_SZ, train_idx, train_sample, train_label, train_dist)
                batch_train, batch_label, batch_dist = batch_train.to(self.device), batch_label.to(self.device), batch_dist.to(self.device)
                self.train_epoch(batch_train, batch_label, batch_dist)

            # valid after trian
            for _iter in range(iter_val_epoch):
                batch_val, batch_label, batch_dist = self.get_batch_data(_iter, BATCH_SZ, val_idx, val_sample, val_label, val_dist)
                batch_val, batch_label, batch_dist = batch_val.to(self.device), batch_label.to(self.device), batch_dist.to(self.device)
                val_epoch_loss, val_accuracy, val_roc_auc = self.val_epoch(batch_val, batch_label, batch_dist)
                val_ave_accuracy += val_roc_auc

            # save the best model
            model_root = osp.join(self.cfig['save_path'], 'models')
            best_model_last = '%s/best_model_last.pth' % (model_root) 
            val_ave_accuracy /= iter_val_epoch
            if val_ave_accuracy > best_val_roc_auc:
                best_val_roc_auc = val_ave_accuracy
                torch.save(self.model.state_dict(), best_model_last)
                print("best model is saved")

    def train_epoch(self, batch_train, batch_label, batch_dist):
        self.model.train()
        self.optim.zero_grad()
        pred = self.model(batch_train, batch_dist)
        pred_prob = torch.sigmoid(pred)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        target = torch.eye(2)[batch_label, :].to(self.device)  # one-hot
        loss = BCE_weight_LOSS(pred, target, count_pos, count_neg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
        self.optim.step()
        
    def val_epoch(self, batch_train, batch_label, batch_dist):
        self.model.eval()
        pred_list, target_list, pos_list = [], [], []
        self.optim.zero_grad()

        pred = self.model(batch_train, batch_dist)  
        pred_prob = torch.sigmoid(pred)
        count_pos = torch.sum(batch_label)
        count_neg = torch.sum(1 - batch_label)
        pos_weight = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)
        target = torch.eye(2)[batch_label, :].to(self.device) # one-hot

        loss = BCE_weight_LOSS(pred, target, count_pos, count_neg)

        pred_cls = pred.data.max(1)[1]
        pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
        pred_list += pred_cls.data.cpu().numpy().tolist()
        target_list += batch_label.data.cpu().numpy().tolist()
        val_epoch_loss = loss.item()
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)
        # print('val accuracy: %.4f   val loss:%.4f     val auc: %.4f' % (OA, val_epoch_loss, roc_auc))
        return val_epoch_loss, OA, roc_auc

    def load_train_data(self, path):
        with open(os.path.join(path, 'train_sample.pickle'), 'rb') as file:
            train_sample = pickle.load(file)
        with open(os.path.join(path, 'train_label.pickle'), 'rb') as file:
            train_label = pickle.load(file)
        with open(os.path.join(path, 'train_dist.pickle'), 'rb') as file:
            train_dist = pickle.load(file)
        return train_sample, train_label, train_dist

    def load_valid_data(self, path):
        with open(os.path.join(path, 'valid_sample.pickle'), 'rb') as file:
            valid_sample = pickle.load(file)
        with open(os.path.join(path, 'valid_label.pickle'), 'rb') as file:
            valid_label = pickle.load(file)
        with open(os.path.join(path, 'valid_dist.pickle'), 'rb') as file:
            valid_dist = pickle.load(file)
        return valid_sample, valid_label, valid_dist

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
    cfig = yaml.load(f) # load Parameter
    trainer = Trainer(cfig)
    trainer.train()
