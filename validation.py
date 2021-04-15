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



R = sp.load_npz("log_rm_mu8_mi2.npz")
with open('temp.txt','w') as file:
	file.write("cross validation on p\n")

ratings={}
ratings['R'] = R


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

	err2,cm_round= B.validation(test, Utilde, V, P=P)
	print("Factorisation RMSE: %f" % (np.sqrt(err)))
	logging.info("Factorisation RMSE: %f" % (np.sqrt(err)))
	print("Prediction RMSE: %f" % (err2))
	logging.info("Prediction RMSE: %f" % (err2))

	acc2=[0,0,0]
	for i in [0,1,2]:
	    acc2[i]=cm_round[i,i]/cm_round.sum(axis=1)[i]

	correctsum = 0
	for i in range(3):
	    correctsum +=cm_round[i,i]

	print(acc2)
	print(cm_round)
	print(cm_round.sum())
	print(correctsum/cm_round.sum())



with open('temp.txt','a') as file:
	file.write(str(acc2)+'\n')
	file.write("confusion matrix:"+'\n')
	file.write(str(cm_round)+'\n')
	file.write(str(cm_round.sum())+'\n')
	file.write("overall accuracy:"+'\n')
	file.write(str(correctsum/cm_round.sum())+'\n')

