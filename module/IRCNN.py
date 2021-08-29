import pickle
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from module import IRCNN_LSTMCell

class IRCNN(nn.Module):
    """
    IRCNN
    
    :param inputs: (T, B, C, H, W)
    T is the length of Time
    B is the Batch size
    C is the Channels
    H is High of the image 
    W is Width of the image 
    
    :param return: outputs:(B, N)
    N is number of classes
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(IRCNN, self).__init__()

        self.dislstmcell = IRCNN_LSTMCell('mode', 64, out_channels, kernel_size, convndim=2)  # mode: LSTM/DisLSTM1/DisLSTM2/DisLSTM3/DisLSTM4
        self.time_len = time_len  # length of time
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=(1,1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=(1,1))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.5)  # 2D Dropout layer
        self.fc1 = nn.Linear(out_channels, 32) # FC layer1
        self.fc2 = nn.Linear(32, num_classes) # FC layer2
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer

    def forward(self, x, time_dis):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.time_len):
            
            t = torch.relu(self.dropout(self.conv1(x[i])))  #(B, in_channels, H, W) --> (B, 16, H, W)

            t = torch.relu(self.dropout(self.conv2(t)))
            t = torch.relu(self.dropout(self.conv3(t)))  #(B, 16, H, W) --> (B, 32, H, W)

            t = torch.relu(self.dropout(self.conv4(t)))
            t = torch.relu(self.dropout(self.conv5(t)))
            t = torch.relu(self.dropout(self.conv5(t)))  #(B, 32, H, W) --> (B, 64, H, W)

            if i == 0:
                hx, cx = self.dislstmcell(t, [time_dis[:, 0], time_dis[:, 0]])
            else:
                hx, cx = self.dislstmcell(t, [time_dis[:, i-1], time_dis[:, i]], (hx, cx))   #(B, 64, H, W) --> (B, out_channels, H, W)

        x = hx.contiguous().view(hx.size()[0], -1)
        x = torch.relu(self.fc1(self.dropout(x))) #(B, out_channels, H, W) --> (B, 32, H, W)
        x = torch.relu(self.fc2(self.dropout(x))) #(B, 32, H, W) --> (B, 2, H, W)
        return x