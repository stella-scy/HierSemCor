import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['KERAS_BACKEND'] = 'theano'

import sys
sys.path.append("../_utils")

import model_msc as model
import pandas as pd


import load as ld
import metrics as metric
import preprocess as prs

import numpy as np
import pickle

from keras import backend as K
from keras.layers import Input, Embedding, GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D, Bidirectional, Dense, dot, Dropout, Multiply,Conv1D,Lambda, Flatten, LSTM, TimeDistributed, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, multiply, Activation


from keras.models import load_model
from keras.models import model_from_json


file_lis = ["data.txt"]

ssm_np = ["ssm_part.npy"]


recon_1 = np.load('recon_1.npy')

elmo_fname = "elmo.txt"
tot_pat_repr = []    

################### DEFINING HYPERPARAMETERS ###################

dimx = 50
dimy = 50
batch_size = 70
vocab_size = 10000
embedding_dim = 1024
nb_filter = 50
filter_length = 5
depth = 1
nb_epoch = 4
num_tensor_slices = 4


alpha = []
pat_reprs = []
pat_reprs_dict = {}
pat_dict = {}
count = 0

####################################################################

for file_l in range(len(file_lis)):
    data1, data2, data3, label_train, train_len, test_len,\
             elmo_model, categ, key = ld.load_mimic(file_lis[file_l], elmo_fname)


    data_l , data_r, data_z, data_c, embedding_matrix, idf_sent_1 = prs.process_data(data1, data2, data3, categ,
                                                 elmo_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)


    X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r,X_train_z,X_test_z,X_dev_z,X_train_c,X_test_c,X_dev_c, idf_train_l, idf_test_l, idf_dev_l = ld.prepare_train_tests(data_l,data_r, data_z, data_c,
                                                                           train_len,test_len, idf_sent_1)

    np_dist = np.load(ssm_np[file_l])
    np_dist = np_dist[0]
    np_dist = np_dist[0:63,0:63]
    ssm_np_train = X_train_r.shape[0]*[np_dist]
    ssm_np_train = np.asarray(ssm_np_train)
    np_shape = np_dist.shape[0]

    pat_dict[count] = [embedding_matrix, X_train_l, X_train_r, X_train_z, X_train_c, ssm_np_train, label_train, recon_1, key, idf_train_l]
    count = count + 1


lrmodel = model.msc
model_name = lrmodel.__name__
lrmodel = lrmodel(pat_dict[0][0], dimx=dimx, dimy=dimy, dimw=np_shape,nb_filter = nb_filter, embedding_dim = embedding_dim,
                  num_slices = num_tensor_slices, filter_length = filter_length, vocab_size = vocab_size, depth = depth)

lrmodel.save_weights('lrmodel_weights.h5')

lrmodel = model.msc
model_name = lrmodel.__name__

for i in range(len(pat_dict)):
    recon_1_n = [pat_dict[i][6]]
    recon_1_n = np.asarray(recon_1_n)

    if i > 0:
        lrmodel.load_weights('lrmodel_weights.h5')
    else:
        lrmodel = lrmodel(pat_dict[i][0], dimx=dimx, dimy=dimy, dimw=np_shape,nb_filter = nb_filter, embedding_dim = embedding_dim,
                num_slices = num_tensor_slices, filter_length = filter_length, vocab_size = vocab_size, depth = depth)

    lrmodel.fit([pat_dict[i][1], pat_dict[i][2], pat_dict[i][3], pat_dict[i][4], pat_dict[i][5], pat_dict[i][9]],[pat_dict[i][6], recon_1_n],batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
    model_weights = lrmodel.get_weights()
    print('lrmodel.summary()', lrmodel.summary())
    layers_m = lrmodel.layers

    for j in range(len(layers_m)):
        name = layers_m[j].name
        wei = layers_m[j].get_weights()

        if name == 'att':
            lis = []
            repres1 = wei[1]
            pat_reprs.append(repres1)
            pat_reprs_dict[pat_dict[i][8]] = repres1


pat_reprs = np.asarray(pat_reprs)

