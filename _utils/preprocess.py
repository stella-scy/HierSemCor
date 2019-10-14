import re
import numpy as np

from nltk import FreqDist
from itertools import chain
from nltk.tokenize import regexp_tokenize

from keras import backend as K
import sys
import pickle

START = '$_START_$'
END = '$_END_$'
unk_token = '$_UNK_$'


def process_data(sent_l,sent_r,sent_z,categ, emb_model=None,dimx=100,dimy=100,vocab_size=10000,embedding_dim=1024):
    with open('idf.pickle', 'rb') as f:
        idf = pickle.load(f)

    sent1 = []
    sent2 = []
    sent3 = []
    sent4 = []
    sent1.extend(sent_l)
    if sent_r:
        sent2.extend(sent_r)
    sent3.extend(categ)
    sent4.extend(sent_z)

    sentence_1 = ["%s %s %s" % (START,x,END) for x in sent1]

    sentence_2 = [[START] + x + [END] for x in sent2]

    sentence_3 = ["%s %s %s" % (START,x,END) for x in sent3]

    sentence_4 = ["%s %s %s" % (START,x,END) for x in sent4]

    tokenize_sent_1 = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence_1]
    tokenize_sent_3 = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence_3]
    tokenize_sent_4 = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence_4]
    
    tokenize_sent = tokenize_sent_1 + sentence_2 + tokenize_sent_3 + tokenize_sent_4
    
    
    freq = FreqDist(chain(*tokenize_sent))
    print('found ',len(freq),' unique words')
    vocab = freq.most_common(vocab_size - 1)
    
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unk_token)
    
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in word_to_index else unk_token for w in sent]
    
    len_train = len(sent_l)
    len_train_2 = len(sent_r)
    len_train_3 = len(sent_z)

    text=[]
    for i in tokenize_sent:
        text.extend(i)
    
    sentences_x = []
    sentences_y = []
    sentences_w = []
    sentences_z = []
    
    for sent in tokenize_sent[0:len_train]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_x.append(temp)

    idf_sent_1 = []
    for lis in sentences_x:
        new_lis = [idf.get(tok, 0.0) for tok in lis]
        idf_sent_1.append(new_lis)
    idf_sent_1 = np.asarray(idf_sent_1)

            
    X_data = []
    for i in sentences_x:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        X_data.append(temp)
    
    X_data = np.array(X_data)
    
    if sent_r:
        for sent in tokenize_sent[len_train:len_train+len_train_2]:
            temp = [START for i in range(dimy)]
            for ind,word in enumerate(sent[0:dimy]):
                temp[ind] = word       
            sentences_y.append(temp)

        y_data=[]
        for i in sentences_y:
            temp = []
            for j in i:
                temp.append(word_to_index[j])
            temp = np.array(temp).T
            y_data.append(temp)
        
        y_data = np.array(y_data)

    for sent in tokenize_sent[len_train+len_train_2:len_train+len_train_2+len_train_3]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_w.append(temp)
            
    w_data = []
    for i in sentences_w:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        w_data.append(temp)

    for sent in tokenize_sent[len_train+len_train_2+len_train_3:]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_z.append(temp)
            
    z_data = []
    for i in sentences_z:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        z_data.append(temp)


    if emb_model:
        embedding_matrix = np.zeros((len(index_to_word) + 1,embedding_dim))
        
        unk = []
        for i,j in enumerate(index_to_word):
            try:
                embedding_matrix[i] = emb_model[j]
            except:
                unk.append(j)
                continue
        print('number of unkown words: ',len(unk))
        print('some unknown words ',unk[0:5])
    
    return X_data,y_data,z_data,w_data,embedding_matrix, idf_sent_1


def loadEmbModel(embfile):
    print('Loading Embedding File.....')
    f = open(embfile)
    model = {}
    for line in f:
        splitline = line.split()
        word = splitline[0]
        embedding = np.array([float(val) for val in splitline[1:]])
        model[word] = embedding
    print('Loaded Embedding Model.....')
    print(len(model), ' words loaded.....')
    return model

def embedding_layer(embedding_matrix,train=False):
    layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix],trainable=train)
    return layer

