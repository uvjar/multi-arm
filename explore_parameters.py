import BLC
import numpy as np
import pandas as pd
import sys
import math
from timeit import default_timer as timer
import scipy.sparse as sp
from scipy import linalg
import argparse
import random
import os.path
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import os

file = '_big' #*change this
clip = True
clip_min = 1 #*change this
clip_max = 10 #*change this
seed = 1
test_ratio = 0.1 #*change this
err_tol = 0.1 
outf = 'Analysis/output'+file+'.csv'

input_params = {
    'p': np.array([8, 16, 64, 128]), 
    'sigma': np.array([1e0, 1e1, 1e2, 1e3]),
    'd': np.linspace(5,25,5).tolist(), 
}

items = sorted(input_params.items())
keys = [item[0] for item in items]
inputs = [item[1] for item in items]
idx = pd.MultiIndex.from_product(inputs, names=keys)
df = pd.DataFrame(np.zeros((len(idx), 3)), columns=["Prediction RMSE", "Factorisation_RMSE", "Time"], index=idx)

with open(outf, "a") as myfile:
    myfile.write("p,sigma,d,Prediction_RMSE,Factorisation_RMSE,Time\n")


for key, row in df.iterrows():
	input("Press Enter to continue (remove me to run batch)...")
	d = np.int(key[0])
	p = np.int(key[1])
	sigma = key[2]
	print('python3 BLC_GPU.py --Rfromfiles --nosave -f '+file+' -t '+str(test_ratio)+' --clip '+str(clip_min)+' '+str(clip_max)+' -seed '+str(seed)+' -p '+str(p)+' -sigma '+str(sigma)+' -d '+str(d)+' -outf '+outf+' -e '+str(err_tol))
	with open(outf, "a") as myfile:
	    myfile.write(str(p)+','+str(sigma)+','+str(d)+',')
	res = os.system('python3 BLC_GPU.py --Rfromfiles --nosave -f '+file+' -t '+str(test_ratio)+' --clip '+str(clip_min)+' '+str(clip_max)+' -seed '+str(seed)+' -p '+str(p)+' -sigma '+str(sigma)+' -d '+str(d)+' -outf '+outf+' -e '+str(err_tol))
	if res != 0:
		with open(outf, "a") as myfile:
			myfile.write('NaN,NaN,NaN\n')
		
	

outdf = pd.read_csv(outf)
outdf.to_html(outf+'.html')
#print(df)