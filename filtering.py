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


with open("urlSet.pkl",'rb') as f:
    urlSet = pickle.load(f)
    urlSet=list(urlSet)
print('urlS loaded, '+str(len(urlS))+' subscribed url in total')


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

urlToWordCount2 = load_dict("urlToWordCount2.pkl")


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



output_path = 'filtered_data/'
urlSetNew=[];
for i,urlIdx in enumerate(f_items):
    url = urlSet[urlIdx]
    urlTail = url[-20:]
    findURL=False
    for k in urlToWordCount2.keys():
        if urlTail in k:
            findURL=True
            break
    if findURL==False:
        continue
    
    
    # get user idx
    user_idx=f_users[i]
    # get url idx
    try:
        url_idx=urlSetNew.index( k)
    except:
        urlSetNew.append(k);
        url_idx=urlSetNew.index(k)
        
    with open(output_path+"users.txt", "a") as user_file:
        user_file.write(str(user_idx)+" ")
    with open(output_path+"items.txt", "a") as item_file:
        item_file.write(str(url_idx)+" ")
    with open(output_path+"ratings.txt", "a") as rating_file:
        rating_file.write(str(f_ratings[i])+" ")  
    with open(output_path+"counts.txt", "a") as count_file:
        count_file.write(str(urlToWordCount2[k])+" ") 

with open(output_path+'urlSet.txt', 'w') as file:
    for fp in urlSetNew:
        file.write(str(fp))
        file.write('\n')