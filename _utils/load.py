import numpy as np
import sys

from dl_text import *

from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
import pickle

################### LOADING, CLEANING AND PROCESSING DATASET ###################
def load_mimic(file_l, model_name, emb_fname):
    
    res_fname = 'test.ref'
    pred_fname = 'pred_%s'%model_name

    pat1_note1 = []
    with open(file_l) as f:
       for l in f:
           pat1_note1.append(l)
    data_train = pat1_note1[0:300]
    data_test = pat1_note1[300:336]
    data_dev = pat1_note1[336:372]
        
    data1_train, data2_train, data3_train, label_train = [], [], [], []
    data1_test, data2_test, data3_test, label_test = [], [], [], []
    data1_dev, data2_dev, data3_dev, label_dev = [], [], [], []
    categ_train, categ_test, categ_dev = [], [], []

    for line1 in data_train:
        line = line1.strip().split("\t")

        data1_train.append(line[1])
        newstr = line[2].replace("[", "")
        newstr = newstr.replace("]", "")
        newstr = newstr.strip()
        newstr = newstr.split(',')
        newstr = [new.replace('"', '') for new in newstr]
        newstr = [new.replace('\'', '') for new in newstr]

        data2_train.append(newstr)
        newstr2 = line[-1].replace("[", "")
        newstr2 = newstr2.replace("]", "")
        newstr2 = newstr2.strip()
        newstr2 = newstr2.split(',')
        newstr2 = [new.replace('"', '') for new in newstr2]
        newstr2 = [new.replace('\'', '') for new in newstr2]
        data3_train.append(newstr2)

        label_train.append(int(line[3]))
        key = int(line[5])
        categ_train.append(line[4])
        
        
    for line1 in data_test:
        line = line1.strip().split("\t")

        data1_test.append(line[1])
        newstr = line[2].replace("[", "")
        newstr = newstr.replace("]", "")
        newstr = newstr.strip()
        newstr = newstr.split(',')
        newstr = [new.replace('"', '') for new in newstr]
        newstr = [new.replace('\'', '') for new in newstr]

        data2_test.append(newstr)
        newstr2 = line[-1].replace("[", "")
        newstr2 = newstr2.replace("]", "")
        newstr2 = newstr2.strip()
        newstr2 = newstr2.split(',')
        newstr2 = [new.replace('"', '') for new in newstr2]
        newstr2 = [new.replace('\'', '') for new in newstr2]
        data3_train.append(newstr2)

        label_test.append(int(line[3]))
        categ_test.append(line[4])

            
    for line1 in data_dev:
        line = line1.strip().split("\t")

        data1_dev.append(line[1])
        newstr = line[2].replace("[", "")
        newstr = newstr.replace("]", "")
        newstr = newstr.strip()
        newstr = newstr.split(',')
        newstr = [new.replace('"', '') for new in newstr]
        newstr = [new.replace('\'', '') for new in newstr]
        data2_dev.append(newstr)

        newstr2 = line[-1].replace("[", "")
        newstr2 = newstr2.replace("]", "")
        newstr2 = newstr2.strip()
        newstr2 = newstr2.split(',')
        newstr2 = [new.replace('"', '') for new in newstr2]
        newstr2 = [new.replace('\'', '') for new in newstr2]
        data3_train.append(newstr2)

        label_dev.append(int(line[3]))
        categ_dev.append(line[4])

    
    data1, data2, data3= [], [], []
    categ = []

    for i in [data1_train, data1_test, data1_dev]:
        data1.extend(i)
    
    for i in [data2_train, data2_test, data2_dev]:
        data2.extend(i)

    for i in [data3_train, data3_test, data3_dev]:
        data3.extend(i)

    for i in [categ_train, categ_test, categ_dev]:
        categ.extend(i)


    train_len = len(data_train)
    test_len = len(data_test)
    
    emb_model = dl.loadEmbModel(emb_fname)
    
    return data1, data2, data3, to_categorical(label_train), to_categorical(label_test), train_len, test_len, emb_model, res_fname, pred_fname, categ, key


####################### SPLITTING DATA TO TRAINING, TEST AND VALIDATION SETS ###################################################
def prepare_train_test(data_l,data_r,data_z,data_c,train_len,test_len, idf_sent_1):
    data_c = [list(data) for data in data_c]
    data_c = np.asarray(data_c)

    X_train_l = data_l[:train_len]
    X_test_l = data_l[train_len:(test_len + train_len)]
    X_dev_l = data_l[(test_len + train_len):]
    
    X_train_r = data_r[:train_len]
    X_test_r = data_r[train_len:(test_len + train_len)]
    X_dev_r = data_r[(test_len + train_len):]

    X_train_z = data_z[:train_len]
    X_test_z = data_z[train_len:(test_len + train_len)]
    X_dev_z = data_z[(test_len + train_len):]

    X_train_c = data_c[:train_len]
    X_test_c = data_c[train_len:(test_len + train_len)]
    X_dev_c = data_c[(test_len + train_len):]

    idf_train_l = idf_sent_1[:train_len]
    idf_test_l = idf_sent_1[train_len:(test_len + train_len)]
    idf_dev_l = idf_sent_1[(test_len + train_len):]

    
    return X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r,X_train_z,X_test_z,X_dev_z,X_train_c,X_test_c,X_dev_c, idf_train_l, idf_test_l, idf_dev_l
