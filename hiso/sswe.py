#coding:utf-8

import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional as F

class Score(nn.Module):
    '''
    全连接层，计算gram语义，情感语义得分
    '''
    def __init__(self):
        super(Score, self).__init__()
        
        self.synt_fc = nn.Sequential(
                nn.Linear(150,128),
                nn.Hardtanh(inplace=True),
                nn.Linear(128,1)
                )

        self.sent_fc = nn.Sequential(
                nn.Linear(150,128),
                nn.Hardtanh(inplace=True),
                nn.Linear(128,2),
                nn.Softmax(dim=-1)
                )

    def forward(self,x):
        synt_score = self.synt_fc(x)
        sent_score = self.sent_fc(x)

        return synt_score, sent_score

class SSWE(nn.Module):
    '''
    SSWE 情感语义模型
    '''
    def __init__(self, opt):
        super(SSWE, self).__init__()
        
        self.opt = opt
        self.lookup = nn.Embedding(opt.voc_size,opt.embed_dim)

        self.score = Score()

    def forward(self,x):
        '''
        x = [left,right, middle, corrput, corrput]
        '''
        x = self.lookup(x)
        true_gram = torch.cat((x[:,0,:],x[:,1,:],x[:,2,:]),dim=1)
        scores = [self.score(true_gram)]

        for step in range(3, x.size(1)):
            corrupt_gram = torch.cat((x[:,0,:],x[:,1,:],x[:,step,:]),dim=1)

            scores.append(self.score(corrupt_gram))
        return scores


class HingeMarginLoss(nn.Module):
    '''
    计算hinge loss 接口
    '''
    def __init__(self):
        super(HingeMarginLoss, self).__init__()

    def forward(self,t,tr,delt=None,size_average=False):
        '''
        计算hingle loss
        '''
        if delt is None:
            loss = torch.clamp(1-t+tr, min=0)
        else:
            loss = torch.clamp(1 - torch.mul(t-tr, torch.squeeze(delt)), min=0)
            loss = torch.unsqueeze(loss,dim=-1)
        # loss = torch.mean(loss) if size_average else torch.sum(loss) 
        return loss

class SSWELoss(nn.Module):
    '''
    SSWE 中考虑gram得分和情感得分的loss类
    '''
    def __init__(self, alpha=0.5):
        super(SSWELoss,self).__init__()
        
        # loss weight for trade-off
        self.alpha = alpha
        self.hingeLoss = HingeMarginLoss()

    def forward(self,scores, labels, size_average=False):
        '''
        [(true_sy_score,true_sem_score), (corrput_sy_score,corrupt_sem_score),...]
        '''
        assert len(scores) >= 2
        true_score = scores[0]
        # all gram have same sentiment hingle loss, because don't use corrupt gram
        sem_loss = self.hingeLoss(true_score[1][:,0],true_score[1][:,1],delt=labels)
        loss = []

        for corpt_i in range(1, len(scores)):
            syn_loss = self.hingeLoss(true_score[0], scores[corpt_i][0])
            cur_loss = syn_loss * self.alpha + sem_loss * self.alpha

            loss.append(cur_loss)

        loss = torch.cat(loss,dim=0)

        if size_average:
            loss = torch.mean(loss)

        return loss


class opt():
    voc_size= 100
    embed_dim=50


if __name__ == '__main__':

    x = Variable(torch.LongTensor([[2,45,34,56,94], [25,76,18,23,56]]))
    y = Variable(torch.FloatTensor([[1], [-1]]))
    
    sswe = SSWE(opt)
    s_loss = SSWELoss()
    scores = sswe(x)
    loss = s_loss(scores, y,size_average=True)
    print(loss.data)
