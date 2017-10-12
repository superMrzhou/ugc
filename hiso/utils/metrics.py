"""
Metrics for multi-label classification.
"""
import numpy as np
from sklearn.metrics import f1_score, label_ranking_loss


def F1_measure(labels, preds, average='binary'):
    '''
    Compute the scores for each output separately
    support 'binary' (default), 'micro', 'macro', 'samples'
    '''
    f1_scores = float(f1_score(labels, preds, average=average))
    return f1_scores


def Hamming_loss(labels, preds):
    '''
    用于度量样本在单个标记上的真实标记和预测标记的错误匹配情况
    @labels: true labels of samples
    @preds:  predict labels of samples
    '''
    hl = float((labels != preds).mean())

    return hl


def One_error(labels, probs):
    '''
    用来考察预测值排在第一位的标记却不隶属于该样本的情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    idx = probs.argsort(axis=1)[:, -1:]
    targets_top_1 = [labels[i, idx[i][0]] for i in range(len(idx))]
    error = 1. - np.mean(targets_top_1)

    return error


def Ranking_loss(labels, probs):
    '''
    用来考察样本的不相关标记的排序低于相关标记的排序情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    return label_ranking_loss(labels, probs)


def Average_precision(labels, probs):
    '''
    用来考察排在隶属于该样本标记之前标记仍属于样本的相关标记集合的情况
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    # 倒序
    prob_sort = np.argsort(-probs, axis=1)
    # generate rank matrix
    for i in range(probs.shape[0]):
        probs[i, prob_sort[i]] = np.arange(1, probs.shape[1] + 1)

    # calcu precision for each sample
    precision_value = 0.
    for i in range(probs.shape[0]):
        n_rank = probs[i][labels[i] == 1]
        n_rank.sort()
        if len(n_rank) == 0: continue
        for idx, rank in enumerate(n_rank):
            precision_value += (idx + 1) / (rank * len(n_rank))
    return precision_value / probs.shape[0]


def Coverage(labels, probs):
    '''
    用于度量平均上需要多少步才能遍历样本所有的相关标记
    @labels: true labels of samples
    @probs:  label's probility  of samples
    '''
    # find min prob of true label
    lbl_probs = labels * probs
    # deal label==0
    lbl_probs[lbl_probs == 0.] = 100
    lbl_probs_min = np.reshape(np.min(lbl_probs, axis=1), (len(probs), -1))
    steps = (probs >= lbl_probs_min).sum(-1).mean()

    return steps


if __name__ == '__main__':

    y_true = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0]])
    y_score = np.array([[0.75, 0.5, 1], [0.1, 0.8, 1], [0.3, 0.7, 0.8]])
    print(y_score >= 0.1)
    print(np.array(y_score >= 0.1, dtype=np.int))
    exit()
    y_preds = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    print('F1@micro: {}'.format(F1_measure(y_true, y_preds, average='micro')))
    print('F1@macro: {}'.format(F1_measure(y_true, y_preds, average='macro')))
    print('hamming_loss: {}'.format(Hamming_loss(y_true, y_preds)))
    print('ranking_loss: {}'.format(Ranking_loss(y_true, y_score)))
    print('one_error: {}'.format(One_error(y_true, y_score)))
    print('coverage: {}'.format(Coverage(y_true, y_score)))
    print('average_precision: {}'.format(Average_precision(y_true, y_score)))
