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


data=R.data
bins=[0.0, 0.15, 0.33];
bins.append(max(data)+1)
print(bins)
cats = pd.cut(data,bins, right=False) 
print(cats.value_counts())
print("Class one percentage ",cats.codes[cats.codes==0].size/cats.codes.size)
print("Class two percentage ",cats.codes[cats.codes==1].size/cats.codes.size)
print("Class three percentage ",cats.codes[cats.codes==2].size/cats.codes.size)

quan_data_5 = cats.codes+1
print(set(quan_data_5)) 
sp.save_npz("filtered_rm_mu8_mi2.npz", R)
