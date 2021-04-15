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



uselog=True
path='filtered_data/'

suffix = ""
if os.path.isfile(path+"counts"+suffix+".npy"):
    f_users = np.load(path+"users"+suffix+".npy")
    print("Reading users... "+str(f_users.size)); sys.stdout.flush()
    f_ratings = np.load(path+"ratings"+suffix+".npy")
    print("Reading ratings... "+str(f_ratings.size)); sys.stdout.flush()
    f_counts = np.load(path+"counts"+suffix+".npy")
    print("Reading counts... "+str(f_counts.size)); sys.stdout.flush()
    f_items = np.load(path+"items"+suffix+".npy")
    print("Reading items... "+str(f_items.size)); sys.stdout.flush()
else:
    f_users = np.loadtxt(open(path+"users"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading users... "+str(f_users.size)); sys.stdout.flush()
    np.save(path+"users"+suffix+".npy",f_users)
    f_ratings = np.loadtxt(open(path+"ratings"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading ratings... "+str(f_ratings.size)); sys.stdout.flush()    
    np.save(path+"ratings"+suffix+".npy",f_ratings)
    f_counts = np.loadtxt(open(path+"counts"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading counts... "+str(f_counts.size)); sys.stdout.flush()    
    np.save(path+"counts"+suffix+".npy",f_counts)
    f_items = np.loadtxt(open(path+"items"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading items... "+str(f_items.size)); sys.stdout.flush()
    np.save(path+"items"+suffix+".npy",f_items)

f_nratings=f_ratings/f_counts
if uselog:
    import math
    
    idx=[]
    for i in range(len(f_nratings)):
        f_nratings[i]=math.log(f_nratings[i])
        if f_nratings[i]>-10:
        	idx.append(i)
    f_nratings=f_nratings[idx]
    f_items=f_items[idx]
    f_users=f_users[idx]
    
    bias=min(f_nratings)
    for i in range(len(f_nratings)):
        f_nratings[i]=f_nratings[i]-bias+0.1        

    min_user=8; min_item=2
    cooR = sp.coo_matrix((f_nratings, (f_items,f_users)), dtype=np.float32);#, shape=(max(f_users)+1, max(f_items)+1)

    R = cooR.tocsc();

    print("Number of rating entries after merging: "+str(R.data.size))
    print(R.shape)
    print("Only keep users who has rated more than "+ str(min_user)+" items")
    R = R[:,np.diff(R.indptr)>min_user];
    print("number of columns(users): "+str(R.indptr.size-1))

    print("Only keep items which has been rated more than "+ str(min_item)+" times")
    print(R.shape)
    R = sp.csr_matrix(R)
    R = R[np.diff(R.indptr)>min_item]
    print("number of rows(items): "+str(R.indptr.size-1))
    print("Got R.")
    print(R.shape)


ratings={}
ratings['R'] = R

# R = sp.load_npz("log_rm_mu8_mi2.npz")

B = BLC.BLC_GPU()
B.p1 = 16
B.test_ratio=0.1;

if B.test_ratio>0:
	train, test = B.split(ratings,B.test_ratio,seed=B.seed)
	print("Training Density: %.5f, ratings: %d, users: %d, items: %d, features: %d, nyms: %d" % (train['R'].nnz/train['R'].shape[0]/train['R'].shape[1], train['R'].nnz, train['R'].shape[0], train['R'].shape[1], B.d, B.p0))
	Utilde, V, err, P = B.run_BLC(train)
	sp.save_npz('tempP.npz', P)
	np.save("tempU.npy", Utilde)
	np.save("tempV.npy", V)
	sp.save_npz("tempR.npz", R)

	err2= B.validation(test, Utilde, V, P=P)
	print("Factorisation RMSE: %f" % (np.sqrt(err)))
	logging.info("Factorisation RMSE: %f" % (np.sqrt(err)))
	print("Prediction RMSE: %f" % (err2))
	logging.info("Prediction RMSE: %f" % (err2))



with open('temp.txt','a') as file:
	file.write(str(acc2)+'\n')
	file.write("confusion matrix:"+'\n')
	file.write(str(cm_round)+'\n')
	file.write(str(cm_round.sum())+'\n')
	file.write("overall accuracy:"+'\n')
	file.write(str(correctsum/cm_round.sum())+'\n')

