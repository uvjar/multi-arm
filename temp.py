import json
from urllib.parse import unquote_plus
import pickle
import os
import sys
import numpy as np
import scipy.sparse as sp
path="data/"
def main():
	print("Retrieving R from files... "); sys.stdout.flush()
	if os.path.isfile(path+"users.npy"):
		f_users = np.load(path+"users.npy")
		print("Got users... "); sys.stdout.flush()
		f_ratings = np.load(path+"ratings.npy")
		print("Got ratings... "); sys.stdout.flush()
		f_movies = np.load(path+"items.npy")
		print("Got items... "); sys.stdout.flush()
	else:
		f_users = np.loadtxt(open(path+"users.txt", "r"), dtype=np.float32)
		print("Got users... "); sys.stdout.flush()
		np.save(path+"users.npy",f_users)
		f_ratings = np.loadtxt(open(path+"ratings.txt", "r"), dtype=np.float32)
		print("Got ratings... "); sys.stdout.flush()
		np.save(path+"ratings.npy",f_ratings)
		f_movies = np.loadtxt(open(path+"items.txt", "r"), dtype=np.float32)
		print("Got items... "); sys.stdout.flush()
		np.save(path+"items.npy",f_movies)

	R=sp.coo_matrix((f_ratings, (f_users, f_movies)), dtype=np.float32);R=sp.csc_matrix(R)
	R = sp.coo_matrix((f_ratings, (f_users, f_movies)), dtype=np.float32)
	R=sp.csc_matrix(R);
	R=R[:,np.diff(R.indptr)>0];
	print("number of columns "+str(R.indptr.size-1))
	R=sp.csr_matrix(R)
	R = R[np.diff(R.indptr)>1]
	R=sp.csr_matrix(R)
	print("number of rows "+str(R.indptr.size-1))
	print("Got R.")

if __name__ == '__main__':
	main()
