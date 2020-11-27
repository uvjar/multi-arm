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

path="data/"
def main():
	print("Retrieving R from files... "); sys.stdout.flush()
	suffix = ""
	if os.path.isfile(path+"users"+suffix+".npy"):
		f_users = np.load(path+"users"+suffix+".npy")
		print("Got users... "+str(f_users.size)); sys.stdout.flush()
		f_ratings = np.load(path+"ratings"+suffix+".npy")
		print("Got ratings... "+str(f_ratings.size)); sys.stdout.flush()
		f_items = np.load(path+"items"+suffix+".npy")
		print("Got items... "+str(f_items.size)); sys.stdout.flush()
	else:
		f_users = np.loadtxt(open(path+"users"+suffix+".txt", "r"), dtype=np.int32)
		print("Got users... "+str(f_users.size)); sys.stdout.flush()
		np.save(path+"users"+suffix+".npy",f_users)
		f_ratings = np.loadtxt(open(path+"ratings"+suffix+".txt", "r"), dtype=np.int32)
		print("Got ratings... "+str(f_ratings.size)); sys.stdout.flush()
		np.save(path+"ratings"+suffix+".npy",f_ratings)
		f_items = np.loadtxt(open(path+"items"+suffix+".txt", "r"), dtype=np.int32)
		print("Got items... "+str(f_items.size)); sys.stdout.flush()
		np.save(path+"items"+suffix+".npy",f_items)

	print("Buliding R...")
	cooR = sp.coo_matrix((f_ratings, (f_users, f_items)), dtype=np.float32);#, shape=(max(f_users)+1, max(f_items)+1)
	print(cooR.data.size)
	R = cooR.tocsc();
	print(R.data.size)
	R = R[:,np.diff(R.indptr)>0];
	print("number of columns "+str(R.indptr.size-1))
	R = sp.csr_matrix(R)
	R = R[np.diff(R.indptr)>5]
	R = sp.csr_matrix(R)
	print("number of rows "+str(R.indptr.size-1))
	print("Got R.")


	data=R.data
	cats = pd.cut(data, [0,40,80,160,600,4676], right=False) 
	print(cats.value_counts())
	print("Class one percentage ",cats.codes[cats.codes==0].size/cats.codes.size)
	print("Class two percentage ",cats.codes[cats.codes==1].size/cats.codes.size)
	print("Class three percentage ",cats.codes[cats.codes==2].size/cats.codes.size)
	print("Class four percentage ",cats.codes[cats.codes==3].size/cats.codes.size)
	print("Class five percentage ",cats.codes[cats.codes==4].size/cats.codes.size)
	quan_data_5 = cats.codes+1
	quan_data_5[quan_data_5==5]=20
	R.data=quan_data_5
	ratingMatrix={}
	ratingMatrix['R'] = R

	B = BLC.BLC()
	B.p1 = 64
	B.test_ratio=0;
	Utilde, V, err, P, cm_round = B.run_BLC(ratingMatrix)
	print(cm_round)

	acc2=[0,0,0,0,0]
	for i in [0,1,2,3,4]:
	    acc2[i]=cm_round[i,i]/cm_round.sum(axis=1)[i]
	print("accuracy for each class",acc2)
	print("average accuracy",sum(acc2)/5)

if __name__ == '__main__':
	main()
