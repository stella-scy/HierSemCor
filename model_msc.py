import keras
import keras as ker
from keras import backend as K
from keras.models import Model
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers.core import Dense, Reshape, Permute
from keras.layers import Input, Embedding, GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D, Bidirectional, Dense, dot, Dropout, Multiply,Conv1D,Lambda, Flatten, LSTM, TimeDistributed, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, multiply, Activation

from keras.layers import Flatten
from keras.layers.merge import concatenate


from keras.layers import Conv2DTranspose

import sys

from numpy.linalg import norm
import numpy as np

from layers import Trilinear, Summ, BiAtten, Mult, LP, SM, G_Score, Unstruct, Fuse, Fuse_2
from keras.layers import Lambda
from keras.layers import multiply

from keras.optimizers import Adam

import layers as lm
import preprocess as prs

######################## MODEL USING BASIC CNN ########################



delta = 0.1
def _smooth_l1(target, output):
    d = target - output
    a = .5 * d**2
    b = delta * (abs(d) - delta / 2.)
    l = K.switch(abs(d) <= delta, a, b)
    return l



def msc(embedding_matrix, dimx=50, dimy=50, dimw=7, nb_filter = 120, num_slices = 3,
        embedding_dim = 1024 ,filter_length = (50,4), vocab_size = 8000, depth = 1):

    print('HierSemCor Model ......')
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    inpz = Input(shape=(dimy,),dtype='int32',name='inpz')
    inps = Input(shape=(dimy,),dtype='int32',name='inpz')
    inpw = Input(shape=(dimw,dimw,),dtype='float32',name='inpw')
    inpi = Input(shape=(dimx,),dtype='float32',name='inpi')

    inpi_c = K.cast(inpi, dtype='int32')

    x = embedding_layer(embedding_matrix,train=True)(inpx)
    y = embedding_layer(embedding_matrix,train=True)(inpy)
    z = embedding_layer(embedding_matrix,train=True)(inpz)
    s = prs.embedding_layer(embedding_matrix,train=True)(inps)

    ''' word-level embeddings: '''
    
    ''' term gating network '''
    idf_conc = concatenate([inpy, inpi_c])
    idf_conc_c = K.cast(idf_conc, dtype='float32')
    g_sc_temp = G_Score(-1)(idf_conc_c)
    g_sc = g_sc_temp[0]

    '''sentence-level embeddings: '''
    
    ''' semantic frame-aware unstructured representation '''
    x_flat = Flatten()(x)
    x_new = Dense(100, activation='relu')(x_flat)
    x_sem_repr = Lambda(lambda x: x[0]*x[1], name="x_sem_repr_temp")([g_sc, x_new])
    x_sem_repr_new = Dense(2500, activation='relu')(x_sem_repr)
    x_sem_repr_fin = Reshape((50,50), name="x_sem_repr")(x_sem_repr_new)

    
    ''' structured representation '''   
    channel_1, channel_2 = [], []

    conv = Conv1D(kernel_size=filter_length, padding="same", activation="relu", data_format="channels_last", filters=nb_filter)

    fr_1 = conv(y)
    sec_1 = conv(z)
    trd_1 = conv(s)
            
    fr_2 = conv(fr_1)
    sec_2 = conv(sec_1)
    trd_2 = conv(trd_1)
    
    conv2 = Conv1D(kernel_size=10, padding="same", activation="relu", data_format="channels_last", filters=50)
    fr_3 = conv2(fr_2)
    sec_3 = conv2(sec_2)
    trd_3 = conv2(trd_2)
        
    merge_temp = concatenate([fr_3, sec_3, trd_3])
    merge = Dense(100, activation='relu')(merge_temp)

    ''' unstructured representation '''

    ssm = Conv1D(kernel_size=(4), padding="same", activation="relu", data_format="channels_last", filters=50)(inpw)

    ssm = Dropout(0.5)(ssm)
    ssm = Permute((2,1))(ssm)
    ssm = Dense(25, activation='relu')(ssm)
    x_m_2 = Dense(25, activation='relu')(x)
    merge_2_temp = concatenate([x_m_2, ssm])
    merge_2 = concatenate([merge_2_temp, x_sem_repr_fin])

    ''' bidirectional trilinear network '''
    att_tot = Trilinear()([merge_2, merge])
    att_a = att_tot[0]
    att_b = att_tot[1]
    comb_att = concatenate([att_a, att_b])
    mlp_1 = Dense(50,activation='relu')(comb_att)
    mlp_2 = Dense(100,activation='relu')(mlp_1)
    soft_comb = Dense(200,activation='tanh',name='att')(mlp_2)
    summed_sc = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name="summed_sc")(soft_comb)

    ''' note representation '''
    doc_repr = Lambda(lambda x: x[0]*x[1], name="doc_repr")([comb_att, soft_comb])
    summed = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name="summed")(doc_repr)
    summed_2 = Lambda(lambda x: K.sum(x, axis=0), name="recon_2")(summed)

    score = Dense(2,name='score')(summed)
    model = Model([inpx, inpy, inpz, inpw, inpi],[score,summed])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile( loss={'score':'categorical_crossentropy', 'summed':_smooth_l1},optimizer=adam) 
    return model

