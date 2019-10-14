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

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist



def write_tf_records(list_c, elmo_model):

    with open('tot_sem_lis.pickle', "rb") as input_file:
        tot_sem_lis = pickle.load(input_file)

    f10p = {k: list_c[k] for k in list(list_c.keys())}
    embedMap = {}
    patEmbedMap = {}
    embed = {}
    c = 0
    totPatLis = []
    for key, val in f10p.items():
        if c == 0:
            patientLis = []
            tot_ssm_list = []
            tot_ssm_list_2 = []
            for v_dict in val:
                ssm_list = []
                ssm_list_2 = []

                count = 0
                subPatLis = []
                for k, (v_cat, v_lis, v_tok) in v_dict.items():
                    for (tup1, tup2) in v_lis:
                        subPatLis.append([count, tup1, tup2, 1, v_cat, key, v_tok])
                        for i in range(5):
                            flag = True
                            while(flag):
                                mySemLis = random.choice(tot_sem_lis)

                                if((mySemLis != tup2) and (len(mySemLis) == len(tup2))):
                                    flag = False
                            count = count + 1
                            subPatLis.append([count, tup1, mySemLis, 0, v_cat, key, v_tok])

                            emp = ['']*(50-len(mySemLis))
                            mySemLis_n = mySemLis + emp
                            embeddings = elmo_model.get_embeddings(mySemLis_n)
                            embeddings = np.reshape(embeddings, (50,3072))
                            embeddings_new = []
                            for tok in embeddings:
                                embeddings_new.append([[x] for x in tok[0:50]])
                            ssm_list.append(embeddings_new)
                            
                            emp2 = ['']*(50-len(v_cat.split()))
                            embeddings_3 = elmo_model.get_embeddings(v_cat.split() + emp2)
                            embeddings_3 = np.reshape(embeddings_3, (50,3072))
                            embeddings_new_3 = []
                            for tok in embeddings_3:
                                embeddings_new_3.append([[x] for x in tok[0:50]])
                            ssm_list_2.append(embeddings_new_3)

                            
                        emp = ['']*(50-len(tup2))
                        tup2 = tup2 + emp
                        embeddings = elmo_model.get_embeddings(tup2)
                        embeddings = np.reshape(embeddings, (50,3072))
                        embeddings_new = []
                        for tok in embeddings:
                            embeddings_new.append([[x] for x in tok[0:50]])
                        ssm_list.append(embeddings_new)
                        emp2 = ['']*(50-len(v_cat.split()))
                        embeddings_3 = elmo_model.get_embeddings(v_cat.split() + emp2)
                        embeddings_3 = np.reshape(embeddings_3, (50,3072))
                        embeddings_new_3 = []
                        for tok in embeddings_3:
                            embeddings_new_3.append([[x] for x in tok[0:50]])
                        ssm_list_2.append(embeddings_new_3)
                        count = count + 1
                    patientLis.append(subPatLis)
                tot_ssm_list.append(ssm_list)
                tot_ssm_list_2.append(ssm_list_2)

            np.save('ssm_tot_1', tot_ssm_list)
            np.save('ssm_tot_2', tot_ssm_list_2)
            tot_ssm_list_2 = np.asarray(tot_ssm_list_2)
            tot_ssm_list = np.asarray(tot_ssm_list)

            totPatLis.append(patientLis)
        c = c + 1



def main():

    with open('data_orig.pickle', "rb") as input_file:
        train_c = pickle.load(input_file)

    elmo_model = ELMO_MIMIC()
    write_tf_records(train_c, elmo_model)


if __name__ == '__main__':
    main()
