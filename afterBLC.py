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


path='filtered_data/'
V = np.load(path+"test1/matrix_V_nym16.npy")
P = sp.load_npz(path+'test1/matrix_P_nym16.npz')
R = sp.load_npz(path+"test1/filtered_rm.npz")
Utilde = np.load(path+"test1/matrix_Utilde_nym16.npy")


def nym_cm(P, Utilde, V, R):
	p = P.shape[0]
	R = R.tocsr() 
	P = P.tocsr()
	cm_round = np.zeros((3,3) ,dtype=int)

	for i in range(p):
		tP = P[i, :]
		if tP.nnz == 0:
			continue

		tR = R[tP.indices, :]
		temp=Utilde[:, i].transpose().dot(V)[tR.indices]

		temp2=np.round(temp)
		temp2[temp2<1]=1; 
		temp2[temp2>3]=3

		for i in range(temp2.shape[0]):
			c=int(temp2[i])-1
			r=int(tR.data[i])-1
			cm_round[r,c]+=1
	return cm_round


cm_round=nym_cm(P, Utilde, V, R)

acc2=[0,0,0]
for i in [0,1,2]:
    acc2[i]=cm_round[i,i]/cm_round.sum(axis=1)[i]
print(acc2)
print(cm_round)

correctsum = 0
for i in range(3):
    correctsum +=cm_round[i,i]
print(cm_round.sum())
print(correctsum/cm_round.sum())


