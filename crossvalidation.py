import json
from urllib.parse import unquote_plus
import pickle
import os
import time as T
import logging
from collections import Counter
import argparse
import sys
import numpy as np
import scipy.sparse as sp
import BLC
import pandas as pd
from sklearn.model_selection import KFold
thresh=3
kf = KFold(n_splits=10, random_state=None, shuffle=True)

# load rating matrix
path='filtered_data/'
R = sp.load_npz("filtered_rm_mu8_mi2.npz")
with open('cv_result_p.txt','w') as file:
	file.write("cross validation on p\n")

import BLC

R = R.tocsr() # make sure R is in csr format
shap=R.shape
R_users=R.nonzero()[0]
R_items = R.nonzero()[1]
for candidate_p in [4,8,16,32,64]:
    print("running candidate "+ str(candidate_p))
    atWhichFold=0
    for train, test in kf.split(range(R.nnz)):
        test_idx=test
        test_R = sp.coo_matrix((R.data[test_idx], (R_users[test_idx], R_items[test_idx])),shape=shap)
        test_R = test_R.tocsr()
        test_R.sort_indices()
        
        train_data = np.delete(R.data, test_idx)
        train_users = np.delete(R_users, test_idx)
        train_items = np.delete(R_items, test_idx)
        train_R = sp.coo_matrix((train_data, (train_users, train_items)),shape=shap)
        train_R = train_R.tocsr()
        train_R.sort_indices()
        # remove users from test data for which 
        #there is insufficient training data 
        #(and so P entry for that user
        # is not determined from training data)
        entries_per_row = train_R.getnnz(axis=1)
        too_few = np.array(range(shap[0]))[entries_per_row<thresh]
        for i in too_few:
            test_R.data[test_R.indptr[i]:test_R.indptr[i + 1]] = 0
        test_R.eliminate_zeros()
        self.vprint("Removed %d rows from test_R due to too few training samples.\n"%too_few.size,"")

        train = {}
        train['R'] = train_R

        test = {}
        test['R'] = test_R
        print("Training Density: %.5f, ratings: %d, users: %d, items: %d, features: %d, nyms: %d" % (train['R'].nnz/train['R'].shape[0]/train['R'].shape[1], train['R'].nnz, train['R'].shape[0], train['R'].shape[1], B.d, B.p0))
        
        B = BLC.BLC_GPU()
        B.p1 = candidate_p
        Utilde, V, err, P = B.run_BLC(train,verbose=0)
        err2,cm_round= B.validation(test, Utilde, V, P=P)
        print("Prediction RMSE: %f" % (err2))

        acc2=[0,0,0]
        for i in [0,1,2]:
            acc2[i]=cm_round[i,i]/cm_round.sum(axis=1)[i]
        correctsum = 0
        for i in range(3):
            correctsum +=cm_round[i,i]
        print("overall accuracy:"+str(correctsum/cm_round.sum()))


        with open('cv_result_p.txt','a') as file:
            file.write("p="+str(candidate_p)+" round="+str(atWhichFold)+"\n")
            file.write(str(acc2)+'\n')
            file.write("confusion matrix:"+'\n')
            file.write(str(cm_round)+'\n')
            file.write(str(cm_round.sum())+'\n')
            file.write("overall accuracy:"+'\n')
            file.write(str(correctsum/cm_round.sum())+'\n')
            file.write("Prediction RMSE: " + str(err2))
        atWhichFold+=1
    