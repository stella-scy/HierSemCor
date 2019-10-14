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
    with open('tot_sem_lis.pickle', "rb") as input_file:
        tot_sem_lis = pickle.load(input_file)

    f10p = {k: list_c[k] for k in list(list_c.keys())}
    embedMap = {}
    patEmbedMap = {}
    embed = {}

    totPatLis = []
    for key, val in f10p.items():

        patientLis = []
        for v_dict in val:
            count = 0
            subPatLis = []
            for k, (v_cat, v_lis, v_tok) in v_dict.items():
                    for i in range(5):
                        flag = True
                        while(flag):
                            mySemLis = random.choice(tot_sem_lis)

                            if((mySemLis != tup2) and (len(mySemLis) == len(tup2))):
                                flag = False
                        count = count + 1
                        subPatLis.append([count, tup1, mySemLis, 0, v_cat, key, v_tok])

                    count = count + 1
                patientLis.append(subPatLis)

                    
        totPatLis.append(patientLis)


    with open('data.pickle', 'wb') as f:
        pickle.dump(totPatLis, f)


def main():

    with open('data_orig.pickle', "rb") as input_file:
        train_c = pickle.load(input_file)


    elmo_model = ELMO_MIMIC()
    write_tf_records(train_c, elmo_model)


if __name__ == '__main__':
    main()
