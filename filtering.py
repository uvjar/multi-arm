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
import pandas as pd

path='data/'
suffix = "_big"
if os.path.isfile(path+"users"+suffix+".npy"):
    f_users = np.load(path+"users"+suffix+".npy")
    print("Reading users... "+str(f_users.size)); sys.stdout.flush()
    f_ratings = np.load(path+"ratings"+suffix+".npy")
    print("Reading ratings... "+str(f_ratings.size)); sys.stdout.flush()
    f_items = np.load(path+"items"+suffix+".npy")
    print("Reading items... "+str(f_items.size)); sys.stdout.flush()
else:
    f_users = np.loadtxt(open(path+"users"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading users... "+str(f_users.size)); sys.stdout.flush()
    np.save(path+"users"+suffix+".npy",f_users)
    f_ratings = np.loadtxt(open(path+"ratings"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading ratings... "+str(f_ratings.size)); sys.stdout.flush()
    np.save(path+"ratings"+suffix+".npy",f_ratings)
    f_items = np.loadtxt(open(path+"items"+suffix+".txt", "r"), dtype=np.int32)
    print("Reading items... "+str(f_items.size)); sys.stdout.flush()
    np.save(path+"items"+suffix+".npy",f_items)

with open("urlSet.pkl",'rb') as f:
    urlSet = pickle.load(f)
    urlSet=list(urlSet)