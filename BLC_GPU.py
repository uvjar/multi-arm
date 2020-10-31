import BLC
import numpy as np
import sys
import math
from timeit import default_timer as timer
import scipy.sparse as sp
import argparse
import os.path
#from sklearn.cross_validation import train_test_split
import warnings
import logging # logging set up
logging.basicConfig(filename='blc.log', level=logging.INFO)
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.nan)

# select GPU to use ...
import os
#os.environ['PYOPENCL_CTX']='1'

mainloop_time = timer()

B = BLC.BLC_GPU()
#B = BLC.BLC()
#B = BLC.BLC_CPU()
ratings = {}

parser = argparse.ArgumentParser(description='Run the BLC algorithm on a set of ratings')
parser.add_argument("--Rfromfiles", help="generate R matrix from files Data/users.txt, Data/movies.txt and Data/ratings.txt, each with an ordered list (otherwise a random R is generated to satisfy a given sparsity -s)", action="store_true")
parser.add_argument("-s", dest="sparsity", metavar="S", help="sparsity to be satisfied if --Rfromfiles not specified (default {})".format(B.sparsity), type=float, default=0.1)
parser.add_argument("-m", help='number of items (movies) to be rated (default {})'.format(B.m), type=int)
parser.add_argument("-n", help='number of users (default {})'.format(B.n), type=int)
parser.add_argument("-p", dest="max_p", metavar="P", help='maximum number of nyms (default {})'.format(B.p1), type=int)
parser.add_argument("-d", help='feature space dimension (default {})'.format(B.d), type=int)
parser.add_argument("-f", dest="file", help='custom ending to default filenames (movies.txt, users.txt, ratings.txt). For movies_small.txt etc., use \'-f _small\'', default='')
parser.add_argument("-e", dest="err", help='error tolerance (default {}). For zero error use \'-e -1\''.format(B.err_tol), type=float)
parser.add_argument("-v", dest="verbosity", help='verbosity of output (default {})'.format(B.verbosity), type=int, default=B.verbosity)
parser.add_argument("-t", dest="test_ratio", help='ratio of test set, put equal to zero to avoid performing the test (default {})'.format(B.test_ratio), type=float, default=B.test_ratio)
parser.add_argument("-sigma", dest="sigma", help='sigma for both U and V (default {})'.format(B.sigU), type=float, default=B.sigU)
parser.add_argument("--clip", nargs=2, help=' clip when computing the test RMSE, min and max required', action="store")
parser.add_argument("--round", help=' round to the nearest integer when computing the test RMSE' , action="store_true")
parser.add_argument("--nosave", help='avoid to save Utilde, V and P' , action="store_true")
parser.add_argument("-seed", dest="seed", help='seed for random split (default {})'.format(B.seed), type=int, default=B.seed)
parser.add_argument("-outf", dest="outputfile", help='output file', default='out.csv')
parser.add_argument("--walsqr",help="Introduce weighting in the ALSQR",action="store_true")

args = parser.parse_args()

if args.walsqr:	B.walsqr = args.walsqr
if args.sparsity: B.sparsity = args.sparsity
if args.m: B.m = args.m
if args.n: B.n = args.n
if args.max_p: B.p1 = args.max_p
if args.d: B.d = args.d
if args.err: B.err_tol = args.err
if args.verbosity != B.verbosity: B.verbosity = args.verbosity
if B.p1 > B.n/B.smallest_nym: B.p1 = math.floor(B.n/B.smallest_nym)
B.test_ratio = args.test_ratio
if args.sigma:
	B.sigU = args.sigma
	B.sigV = args.sigma
B.round = args.round
if args.clip:
	B.clip = True
	B.clip_min = np.float(args.clip[0])
	B.clip_max = np.float(args.clip[1])
B.seed = args.seed

# Start parsing R
if args.Rfromfiles:
	print("Retrieving R from files... ", end=""); sys.stdout.flush()

	if os.path.isfile("Data/users"+args.file+".npy"):
	# if args.file == "_netflix":
		f_users = np.load("Data/users"+args.file+".npy")
		print("Got users... ", end=""); sys.stdout.flush()
		f_ratings = np.load("Data/ratings"+args.file+".npy")
		print("Got ratings... ", end=""); sys.stdout.flush()
		f_movies = np.load("Data/movies"+args.file+".npy")
		print("Got movies... ", end=""); sys.stdout.flush()
	else:
		f_users = np.loadtxt(open("Data/users"+args.file+".txt", "r"), dtype=np.float32)
		print("Got users... ", end=""); sys.stdout.flush()
		np.save("Data/users"+args.file+".npy",f_users)
		f_ratings = np.loadtxt(open("Data/ratings"+args.file+".txt", "r"), dtype=np.float32)
		print("Got ratings... ", end=""); sys.stdout.flush()
		np.save("Data/ratings"+args.file+".npy",f_ratings)
		f_movies = np.loadtxt(open("Data/movies"+args.file+".txt", "r"), dtype=np.float32)
		print("Got movies... ", end=""); sys.stdout.flush()
		np.save("Data/movies"+args.file+".npy",f_movies)

	ratings['R'] = sp.coo_matrix((f_ratings, (f_users, f_movies)), dtype=np.float32)

	print("Got R.")
else:
	# Generate R to a given sparsity
	print("Generating random R...")
	#ratings['R'] = BLC.init_R(B.p1, B.n, B.m, B.d, B.sparsity)
	B.p1 = 4
	B.d = 5
	B.m = 100
	B.n = 1000
	# add noise with different variance to each item (column) of R
	Rvar = np.diag(range(0,100))/100
	noise = np.random.randn(1000,100).dot(Rvar)
	ratings['R'] = BLC.init_R(B.p1, B.n, B.m, B.d, 0.1, noise)
	B.test_ratio = 0.1

	# user weighted least squares ...
	B.walsqr = True

	# Add noise
	Rvar = np.diag(range(0,B.m))/B.m
	Noise = np.random.randn(B.n,B.m).dot(Rvar)  # add noise with different variance to each item (column) of R
	ratings['R'] = ratings['R'] + Noise
	# Add side information for users and items
	print("Got R.")

# FOR LASSO UNCOMMENT!
#B.L2=-1 # use L1 penalty
#B.alpha=0.015

B.p1 = 16
B.walsqr = True
# full netflix values ...
#B.d=20
#B.p1=128
ratings,original_rows,original_columns = BLC.prepare_R(ratings)
print("R Density: %.5f, ratings: %d, users: %d, items: %d, features: %d, nyms: %d" % (ratings['R'].nnz/ratings['R'].shape[0]/ratings['R'].shape[1], ratings['R'].nnz, ratings['R'].shape[0], ratings['R'].shape[1], B.d, B.p0))
if B.test_ratio>0:
	train, test = B.split(ratings,B.test_ratio,seed=B.seed)
	# print(train,test)
	print("Training Density: %.5f, ratings: %d, users: %d, items: %d, features: %d, nyms: %d" % (train['R'].nnz/train['R'].shape[0]/train['R'].shape[1], train['R'].nnz, train['R'].shape[0], train['R'].shape[1], B.d, B.p0))

print("Pre-processing time: %.3f\n" % (timer()-mainloop_time))
after_proc_time = timer()

logging.info("\n\nRunning weighted alternating least squares? %r\n" % (args.walsqr))
print("Running weighted alternating least squares? %r\n" % (args.walsqr))
if B.test_ratio>0:
	Utilde, V, err, P = B.run_BLC(train)
	err2 = B.validation(test, Utilde, V, P=P)
	print("Factorisation RMSE: %f" % (np.sqrt(err)))
	logging.info("Factorisation RMSE: %f" % (np.sqrt(err)))
	print("Prediction RMSE: %f" % (err2))
	logging.info("Prediction RMSE: %f" % (err2))
	#U_MF, V_MF, err_MF = B.run_MF(train)
	#err2_MF = B.validation(test, U_MF, V_MF, P=sp.eye(U_MF.shape[1]))
	#print("Factorisation RMSE MF: %f" % (np.sqrt(err_MF)))
	#print("Prediction RMSE for MF: %f" % (err2_MF))
else:
	Utilde, V, err, P = B.run_BLC(ratings)
	err2 = None
	#U_MF, V_MF, err_MF = B.run_MF(ratings)
	#err2_MF = None

#print("NAIVE",B.naive(train,test))

# calculate variance and lambda to store in files ...
R = ratings['R']
B.varbound = False
Rvar = B.variance(P,R).toarray()
lam = P.dot(R.astype(bool)).toarray()

with open(args.outputfile, "a") as myfile:
    myfile.write(str(err2)+","+str(err)+","+str(timer()-after_proc_time)+"\n")

# save matrices in matlab compatible text format
if not args.nosave:
	np.savetxt('Utilde', Utilde, delimiter=',')
	np.savetxt('V', V, delimiter=',')
	np.savetxt('P', np.vstack(P.T.nonzero()).T, fmt='%d', delimiter=',') # user, nym pairs, one per line
	np.savetxt('Rvar', Rvar,delimiter=',')
	np.savetxt('lam', lam,delimiter=',')
	np.save('Utilde.npy',Utilde)
	np.save('V.npy',V)
	np.save('P.npy',P.todense())
	np.save('Rvar.npy', Rvar)
	np.save('lam.npy', lam)
	np.save('rows.npy',original_rows)
	np.save('columns.npy',original_columns)

# sample_size = 100000

# R = R.tocoo()

# print("\nStarting error analysis on samples of %d" % (sample_size))

# errors = np.array([], dtype=np.float32)

# for i in range(1000):
# 	sample = random.sample(range(Rnnz), sample_size)
# 	f_ratings = R.data[sample]
# 	f_users = R.row[sample]
# 	f_movies = R.col[sample]
# 	sR = sp.coo_matrix((f_ratings, (f_users, f_movies)), dtype=int, shape=(n, m))
# 	sR = sR.tocsr()

# 	err_total = 0.0

# 	for i in range(p):
# 		tP = P[i,]
# 		if tP.nnz == 0:
# 			continue

# 		tR = sR[tP.toarray().flatten()>0, :]

# 		if tR.nnz == 0:
# 			continue;

# 		err_total += np.square(tR[tR.nonzero()]-Utilde[:, i].transpose().dot(V)[tR.nonzero()[1]]).sum()

# 	errors = np.append(errors, err_total/sR.nnz)

# fl = open("errors_%d.txt" % (d), "w")
# errors = errors.astype(str)
# fl.write("\n".join(errors))

# print("Error analysis done!")
