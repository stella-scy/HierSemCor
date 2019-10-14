import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints , activations

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
from keras.models import Sequential


np.random.seed(21)


class G_Score(Layer):
    
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(G_Score, self).__init__(**kwargs)

    def build(self,input_shape):
        super(G_Score, self).build(input_shape)

    def call(self, data, mask=None):
        g_score = ker.activations.softmax(data, self.axis)

        return g_score


class Unstruct(Layer):
    
    def __init__(self,  **kwargs):
        super(Unstruct, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Unstruct, self).build(input_shape)

    def call(self, data, mask=None):
        x_sem_repr_fin = data[1]*1
        merge_2 = concatenate([data[0], x_sem_repr_fin])

        return merge_2

class Fuse(Layer):
    
    def __init__(self,  **kwargs):
        super(Fuse, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Fuse, self).build(input_shape)

    def call(self, data, mask=None):
        summed_sc = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name="summed_sc")(data[0])
        doc_repr = Lambda(lambda x: x[0]*x[1], name="doc_repr")([data[1], data[0]])
        summed = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]), name="summed")(doc_repr)
        return summed


class Fuse_2(Layer):
    
    def __init__(self,  **kwargs):
        super(Fuse_2, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Fuse_2, self).build(input_shape)

    def call(self, data, mask=None):
        summed_2 = Lambda(lambda x: K.sum(x, axis=0), name="recon_2")(data)
        return summed_2



class BiAtten(Layer):
    
    def __init__(self, shap, **kwargs):
        self.shap = shap
        super(BiAtten, self).__init__(**kwargs)

    def build(self,input_shape):
        self.W1 = self.add_weight(name='W1',shape=self.shap, initializer='random_normal', trainable=True )
        super(BiAtten, self).build(input_shape)

    def call(self, data, mask=None):
        proj_comb = K.batch_dot(self.W1, data)
        return proj_comb

class Summ(Layer):
    
    def __init__(self, axis, keepdims, **kwargs):
        self.axis = axis
        self.keepdims = keepdims
        super(Summ, self).__init__(**kwargs)

    def build(self,input_shape):                                
        super(Summ, self).build(input_shape)

    def call(self, data, mask=None):
        summy = K.sum(data, self.axis, self.keepdims)
        return summy


class Mult(Layer):
    
    def __init__(self, **kwargs):
        super(Mult, self).__init__(**kwargs)

    def build(self,input_shape):                                
        super(Mult, self).build(input_shape)

    def call(self, data, mask=None):
        mul = data[0]*data[1]
        return mul

class LP(Layer):
    
    def __init__(self, **kwargs):
        super(LP, self).__init__(**kwargs)

    def build(self,input_shape):                                
        super(LP, self).build(input_shape)

    def call(self, data, mask=None):
        model = Sequential(name="lp")
        model.add(Dense(200,activation='tanh',name='att'))
        return model

class SM(Layer):
    
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SM, self).__init__(**kwargs)

    def build(self,input_shape):
        super(SM, self).build(input_shape)

    def call(self, data, mask=None):
        weight = ker.activations.softmax(data, axis=self.axis)
        return weight

class Expand(Layer):
    
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(Expand, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Expand, self).build(input_shape)

    def call(self, data, mask=None):
        new = K.expand_dims(data, axis=self.axis)
        return new


class SQ(Layer):
    
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SQ, self).__init__(**kwargs)

    def build(self,input_shape):
        super(SQ, self).build(input_shape)

    def call(self, data, mask=None):
        neww = K.squeeze(data, axis=self.axis)
        return neww


class Trilinear(Layer):
    
    def __init__(self, **kwargs):
        super(Trilinear, self).__init__(**kwargs)

    def build(self,input_shape):
        super(Trilinear, self).build(input_shape)

    def call(self, data, mask=None):
        uns_expand = K.tile(K.expand_dims(data[0], 2), [1, 1, 50, 1])
        st_expand = K.tile(K.expand_dims(data[1], 1), [1, 50, 1, 1])
        mat = concatenate([st_expand, uns_expand, multiply([st_expand,uns_expand])], axis=3)
        similarity = Dense(1)(mat)
        similarity = K.squeeze(similarity,axis=3)
        similarity_row_normalized = ker.activations.softmax(similarity, axis=1)
        similarity_column_normalized = ker.activations.softmax(similarity, axis=2)
        matrix_a = K.batch_dot(similarity_row_normalized, data[1])
        scn = Permute((2,1))(similarity_column_normalized)
        matrix_b = K.batch_dot(K.batch_dot(similarity_row_normalized, scn), data[0])
        return [matrix_a, matrix_b]


class ntn(Layer):
    def __init__(self,inp_size, out_size, p_size=1, activation='tanh', **kwargs):
        super(ntn, self).__init__(**kwargs)
        self.k = out_size
        self.d = inp_size
        self.p = p_size
        self.activation = activations.get(activation)
        self.test_out = 0
        
    def build(self,input_shape):
        
        self.S = self.add_weight(name='s',shape=(self.k, self.d, self.p),
                                      initializer='glorot_uniform',trainable=True)
        self.T = self.add_weight(name='t',shape=(self.k, self.p, self.d),
                                      initializer='glorot_uniform',trainable=True)
        
        self.V = self.add_weight(name='v',shape=(self.k, self.d*2),
                                      initializer='glorot_uniform',trainable=True)
        self.U = self.add_weight(name='u',shape=(self.k,1),
                                      initializer='ones',trainable=False)
                initializer='glorot_uniform',trainable=True)
        super(ntn, self).build(input_shape)     

    def call(self , x, mask=None):
        
        e1=x[0].T
        e2=x[1].T

        
        batch_size = K.shape(x[0])[0]
        sim = []
        V_out = K.dot(self.V, K.concatenate([e1,e2],axis=0))

        for i in range(self.k): 
            temp = K.batch_dot(K.dot(e1.T,K.dot(self.S[i,:,:],self.T[i,:,:])),e2.T,axes=1)
            sim.append(temp)

        sim=K.reshape(sim,(self.k,batch_size))

        tensor_bi_product = self.activation(V_out+sim)
        tensor_bi_product = K.dot(self.U.T, tensor_bi_product).T

        return tensor_bi_product
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)      
    

class Abs(Layer):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        inp1, inp2 = x[0],x[1]
        return K.abs(inp1-inp2)
    
    def get_output_shape_for(self, input_shape):
        return input_shape

class Exp(Layer):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)

    def call(self, x, mask=None):
        h1 = x[0]
        h2 = x[1]    
        dif = K.sum(K.abs(h1-h2),axis=1)  
        h = K.exp(-dif)
        h=K.clip(h,1e-7,1.0-1e-7)
        h = K.reshape(h, (h.shape[0],1))
        return h

    def compute_output_shape(self, input_shape):
        out_shape  = list(input_shape[0])
        out_shape[-1] = 1
        return tuple(out_shape)

def mse(y_true, y_pred):
    y_true=K.clip(y_true,1e-7,1.0-1e-7)    
    return K.mean(K.square(y_pred - y_true), axis=-1)
