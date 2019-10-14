import os
import tensorflow as tf
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm

from clinical_concept_extraction.elmo_vector import ELMO_MIMIC
from html import unescape
import sys

from scipy import sparse
import numpy as np
import random
import pandas as pd


def write_tf_records(list_c, elmo_model):
    embed = {}
    for key, val in list_c.items():

        for v_dict in val:
            for k, (v_cat, v_lis) in v_dict.items():
                for (tup1, tup2) in v_lis:
                    embeddings = elmo_model.get_embeddings(tup1.split())
                    embeddings_2 = elmo_model.get_embeddings(tup2)
                    embeddings_3 = elmo_model.get_embeddings(v_cat.split())
                    float_list = [tf.train.FloatList(value=embedding.flatten().tolist()) for embedding in embeddings]
                    float_list_2 = [tf.train.FloatList(value=embedding.flatten().tolist()) for embedding in embeddings_2]
                    float_list_3 = [tf.train.FloatList(value=embedding.flatten().tolist()) for embedding in embeddings_3]
                    for i, tup in enumerate(tup1.split()):
                        if tup in embed:
                            pass
                        else:
                            embed[tup] = float_list[i].value[0:50]
                    for i, tup in enumerate(tup2):
                        if tup in embed:
                            pass
                        else:
                            embed[tup] = float_list_2[i].value[0:50]

                    for i, tup in enumerate(v_cat.split()):
                        if tup in embed:
                            pass
                        else:
                            embed[tup] = float_list_3[i].value[0:50]
    
    data = pd.DataFrame.from_dict(embed, orient='index')
    data.to_csv('sem_embed_total.txt', sep=' ', header=False, encoding='utf-8')



def main():
    with open('data_orig.pickle', "rb") as input_file:
        train_c = pickle.load(input_file)

    elmo_model = ELMO_MIMIC()
    write_tf_records(train_c, elmo_model)

if __name__ == '__main__':
    main()
