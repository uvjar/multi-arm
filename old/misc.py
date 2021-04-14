import json
import pickle
import os
import time as T
import logging
import sys
import numpy as np
import argparse
import scipy.sparse as sp
logging.basicConfig(filename="test.log", level=logging.DEBUG)
   


# The activeTime for each article the user read throughout the 3 month is normalized by the largest activeTime of each user
def normalizeActiveTime(userevt):
    for usr in userevt:
        total=[]
        for evt in userevt[usr]:
            total.append(evt['activeTime'])
        for evt in userevt[usr]:
            evt['activeTime'] = float(evt['activeTime'])/max(total)
    return userevt
    
def filterRow(n,usr_url_matrix):
    usr_url_matrix_filter_C_R = usr_url_matrix[((usr_url_matrix!=0).sum(axis=1)>n),:]
    return usr_url_matrix_filter_C_R


def main():#user_url_mat.npy
    parser = argparse.ArgumentParser(description='Run the BLC algorithm on a set of ratings')
    parser.add_argument("-f", type=str)
    parser.add_argument("-n",help='least nonempty items in a row',type=int)
    args = parser.parse_args()
    if os.path.isfile(args.f):
        user_url_mat=np.load(args.f)
        usr_url_matrix_filter_C_R = filterRow(args.n, user_url_mat);
        print("matrix shape ");print(usr_url_matrix_filter_C_R.shape);
        np.save("usr_url_matrix_R_"+str(args.n)+".npy",usr_url_matrix_filter_C_R)
    else: 
        print("File does not exist")

        
if __name__=='__main__':
    main()

