# -*- coding:utf-8 -*-
#
# import sys
# import re
# reload(sys)
# sys.setdefaultencoding('utf8')
#
# with open('/Users/Kevin/Documents/Project/sohu/adios/docs/CNN/corpus_jiusuan.txt','rb') as f, \
#      open('/Users/Kevin/Documents/Project/sohu/adios/docs/CNN/corpus_jiusuan_all.txt','wb') as fa, \
#      open('/Users/Kevin/Documents/Project/sohu/adios/docs/CNN/corpus_jiusuan_sample.txt','wb') as fs:
#     i = 0
#     for line in f:
#         cont = line.decode('gb18030').encode('utf-8')
#         s = re.search(r'。([^。]*?就算[^。]*?。)',cont)
#         if s and s.groups()[0]:
#             fa.write('%s\n'%s.groups()[0])
#             i+=1
#         if s and i % 8 == 0:
#             fs.write('%s\n'%s.groups()[0])
#
#  fastXML data_helper

import numpy as np
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

def y2list(y):

    y = [yy[0].strip('\n').split('&') for yy in y]
    return [list(set([re.split('-|_', lbl)[0] for lbl in yy])) + yy for yy in y]


def get_Y0_and_Y1(file_path):
    with open(file_path, 'r') as f_cate:
        Y0, Y1 = [], []
        for line in f_cate:
            if re.search('-|_', line.split('\t')[0]):
                Y0.append(line.strip('\n').split('\t')[1])
            else:
                Y1.append(line.strip('\n').split('\t')[1])
    return Y0, Y1


def filter_data(x, y):
    res_x, res_y = [], []
    for i, yy in enumerate(y):
        temp_y = filter(lambda lbl: '其他' not in lbl and '新闻' not in lbl, yy)
        if temp_y:
            res_x.append(x[i])
            res_y.append(temp_y)
        if i % 2000==0:
            print i
    return res_x, res_y
def load_data_and_labels(file_path, split_tag='\t', lbl_text_index=[0, 1], is_shuffle=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    raw_data = list(open(file_path, 'r').readlines())
    # parse label
    labels = [data.strip('\n').split(split_tag)[lbl_text_index[0]]
              for data in raw_data]
    # parse text
    texts = [data.strip('\n').split(split_tag)[lbl_text_index[1]]
             for data in raw_data]

    # Split by words
    # texts = [clean_str(sent) for sent in texts]
    texts = [filter(lambda a:a !='',s.split(" ")) for s in texts]
    # support multi-label
    labels = [filter(lambda a:a !='',s.split(" ")) for s in labels]
    if is_shuffle:
        ind = np.arange(len(texts))
        np.random.shuffle(ind)
        texts = list(np.array(texts)[ind])
        labels = list(np.array(labels)[ind])
    return texts, labels
if __name__ == '__main__':
    # load data
    print 'loading data.....'
    texts,labels = load_data_and_labels('../docs/CNN/imageText_ml_v5',split_tag='@@@',lbl_text_index=[0,1])

    # calcu tf-idf weights
    vectorizer = TfidfVectorizer(min_df=1)
    print 'transfoming tfifd vector .....'
    texts_vec = vectorizer.fit_transform([' '.join(x) for x in texts])
    vecs = []
    print 'deal to xml format text......'
    cnt = 0
    for i in texts_vec:
        w_ids = i.indices
        weights = i[:,w_ids].data[::-1]
        vec_line = ' '.join(['%s:%.4f'%(w_ids[ii],weights[ii]) for ii in range(len(w_ids))])
        vecs.append(vec_line)
        cnt += 1
        if cnt % 2000 == 0:
            print cnt
    # labels Prepare
    labels = y2list(labels)
    print 'start filter labels....'
    vecs,labels = filter_data(vecs,labels)
    # cate
    lbl_counts = Counter(itertools.chain(*labels))
    # Mapping from index to word
    cate = [x[0] for x in lbl_counts.most_common()]
    cate_id = {v:i for i,v in enumerate(cate)}
    # xml labels
    print 'deal to xml format labels.....'
    xml_lbls = []
    for i,lbls in enumerate(labels):
        xml_lbls.append(' '.join(['%s:1'%cate_id[lbl] for lbl in lbls]))
        if i %2000==0:
            print i

    # shuffle
    print 'shuffle and split .....'
    ind = np.arange(len(vecs))
    np.random.shuffle(ind)

    # split train and test
    tst_n = int(0.2*len(vecs))
    with open('../docs/CNN/xml_train_x_a','w') as ftx,\
         open('../docs/CNN/xml_train_y_a','w') as fty:
         ftx.write('%s %s\n'%(len(vecs) - tst_n,len(vectorizer.vocabulary_)))
         fty.write('%s %s\n'%(len(vecs) - tst_n,len(cate_id)))

         lbl_temp = np.array(xml_lbls)[ind[tst_n:]]
         for i,txt in enumerate(np.array(vecs)[ind[tst_n:]]):
             ftx.write('%s\n'%txt)
             fty.write('%s\n'%lbl_temp[i])
             if i%2000 == 0:
                 print i

    with open('../docs/CNN/xml_test_x_a','w') as ftx,\
         open('../docs/CNN/xml_test_y_a','w') as fty:
         ftx.write('%s %s\n'%(tst_n,len(vectorizer.vocabulary_)))
         fty.write('%s %s\n'%(tst_n,len(cate_id)))

         lbl_temp = np.array(xml_lbls)[ind[:tst_n]]
         for i,txt in enumerate(np.array(vecs)[ind[:tst_n]]):
             ftx.write('%s\n'%txt)
             fty.write('%s\n'%lbl_temp[i])
             if i  % 2000 ==0:
                 print i
