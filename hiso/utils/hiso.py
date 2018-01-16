#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/6 11:13
# @From    : PyCharm
# @File    : hiso
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F

class HISO(nn.Module):
    def __init__(self, opt):
        super(HISO, self).__init__()

        self.model_name = 'HISO'
        self.opt = opt
        # Embedding Layer
        self.wd_embed = nn.Embedding(opt.voc_size, opt.embed_dim)
        self.pos_embed = nn.Embedding(opt.pos_size, opt.embed_dim)

        # Bi-GRU Layer
        self.wd_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True,
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )

        # Bi-GRU Layer
        self.pos_bi_gru = nn.GRU(input_size = opt.embed_dim,
                hidden_size = opt.ghid_size,
                num_layers = opt.glayer,
                bias = True, 
                batch_first = True,
                dropout = 0.5,
                bidirectional = True
                )
        # output from pos hidden layer to predict middle labels
        pos_hidden_size = opt.ghid_size * 2
        self.pos_fc = nn.Sequential(
                nn.BatchNorm1d(pos_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(pos_hidden_size, opt.auxiliary_labels),
                nn.Softmax(dim=-1)
                )
        # predict final labels
        combine_size = opt.ghid_size * 2 + opt.auxiliary_labels
        self.fc = nn.Sequential(
                nn.Linear(combine_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),
                nn.Linear(128, opt.label_dim),
                nn.Softmax(dim=-1)
                )
        

    def forward(self, wd, pos):
        # encoder
        wd = self.wd_embed(wd)
        pos = self.pos_embed(pos)
        # Bi-GRU
        h0 = self.init_hidden(wd.size()[0])
        wd_out, wd_hidden = self.wd_bi_gru(wd)

        pos_out, pos_hidden = self.pos_bi_gru(pos)
        
        # pos_out to predict auxiliary label
        auxi_probs = self.pos_fc(pos_out[:, -1, :])

        # combine wd_out with auxi_probs as feature
        combine_feature = torch.cat((wd_out[:, -1, :], auxi_probs), dim=1)
        logits = self.fc(combine_feature)

        return logits, auxi_probs

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.opt.glayer, batch_size, self.opt.ghid_size)
        return Variable(h0)

class HisoLoss(nn.Module):
    def __init__(self, opt):
        super(HisoLoss, self).__init__()
        self.opt = opt
        # self.reconstruction_loss = 0 // todo

    def forward(self,auxi_probs, auxi_labels, final_probs, final_labels):
        # calcu auxi_labels margin loss
        self.auxi_loss = self.marginLoss(auxi_probs, auxi_labels)

        # calcu final_labels margin loss
        self.auxi_loss = self.marginLoss(final_probs, final_labels) 

    def marginLoss(self, probs, labels):
        
        left = F.relu(self.opt.max_margin - probs, inplace=True)**2
        right = F.relu(self.opt.min_margin - probs, inplace=True)**2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        return margin_loss.sum()


class opt(object):
    voc_size = 100
    pos_size = 100
    embed_dim = 50
    ghid_size = 18
    seq_len = 4
    glayer = 2
    auxiliary_labels = 3
    label_dim = 6

if __name__ == '__main__':
    import visdom
    
    wd = Variable(torch.LongTensor([[2,45,75,34], [5,54,76,23]]))
    pos = Variable(torch.LongTensor([[73,45,87,2], [13,56,7,43]]))
    model = HISO(opt)
    outputs = model(wd,pos)
    print(outputs)
