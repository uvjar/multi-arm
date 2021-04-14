# BLC v0.2

from numpy.linalg import inv
from scipy import linalg
import numpy as np
import scipy.sparse as sp
import multiprocessing
from timeit import default_timer as timer
from functools import partial
import pyopencl as cl
import math
import sys
from sklearn.linear_model import ElasticNet
from numpy.lib.stride_tricks import as_strided
#import logging # logging set up
#logging.basicConfig(filename='netflix_error.log', level=logging.INFO)
#np.set_printoptions(threshold=np.nan)

# todo:
# 1. can we map back to the movie ids and user ids?

class BLC:
	def __init__(self):
		self.n = 500						# Number of users
		self.m = 250						# Number of items
		self.d = 10							# Dimension of the latent feature space
		self.p0 = 1							# Starting number of nyms
		self.p1 = 2**5						# Maximum number of nyms
		self.sig = 0.1						# Standard deviations
		self.sigU = 1e2
		self.sigV = 1e2
		self.eps = 1e-3					# Tolerance for convergence
		self.Ptol = 0.01				# Ratio of users changing nyms before considered converged
		self.err_tol = 0.01			# Tolerange for MSE
		self.log_frequency = 10	# Base frequency by which alsqr is run (at frequency, frequency**2, frequency**3, etc, changes)
		self.max_iter_split = 1000
		self.max_iter_p = 10			# Maximum number of iterations until P converges
		self.max_iter_alsqr = 5			# Maximum number of iterations until Utilde and V converge
		self.verbosity = 1			# Verbose output?
		self.smallest_nym = 1		# Minimum number of users allowed per nym
		self.p_validate = 5
		self.nym_split_noise = 0.0001
		self.L2 = 1
		self.alpha=0.001
		self.test_ratio = 0.1
		self.clip = False
		self.clip_min = None
		self.clip_max = None
		self.round = False
		self.seed = None
		self.max_time_per_split = 600 # ten minutes per split
		self.sparsity = 0.1
		self.version = "0.2"
		self.walsqr = False # not running weighted least squares
		self.varbound = True # Upper and lower bounds in variance function, taken off for final calc.

	# Log-likelihood function
	def ptilde(self, Rtilde, Utilde, V):
		# when Rtilde is sparse then Rtilde[Rtilde.nonzero()] - Utilde.T.dot(V)[Rtilde.nonzero()] is a matrix
		# whereas when Rtilde is an ndarray then it is an ndarray, so we use np.asarray() to make sure its
		# an ndarray we have in err and so **2 works as expected (applied to a matrix it gives the matrix exponential)
		err = np.asarray(Rtilde[Rtilde.nonzero()] - Utilde.T.dot(V)[Rtilde.nonzero()])
		return (err**2).sum()
		#return ((Rtilde[Rtilde>0] - Utilde.T.dot(V)[Rtilde>0]) ** 2).sum() # is this faster or slower than the above ?

	# Re-calculate lam matrix of vectors, based on R and the updated P
	def recalc_lam_Rtilde(self, P, R, Ridx):
		lam = P.dot(Ridx).toarray()
		lamInv = np.zeros_like(lam, dtype=float)
		lamInv[lam>0] = 1/lam[lam>0]
		Rtilde = P.dot(R).toarray()*lamInv

		if self.walsqr == True: # check if running weighted alsqr
			Rvar_inv = self.inv_variance(P, R)
			lam = np.multiply(lam,Rvar_inv)

		return lam, Rtilde

	# Execute alternating least squares, and display results
	def run_alsqr(self, Utilde, V, lam, Rtilde):
		itr = 0
		E = 1e10; dl = 1e10
		tocU=0.0; tocV = 0.0
		while ((E > self.eps) and (dl > self.eps) and (itr < self.max_iter_alsqr)):
		#while ((E > self.eps) and (itr < self.max_iter_alsqr)):
			# Calculate Utilde
			ticU = timer()
			Utilde = self.alsqr_U(V, lam, Rtilde, Utilde)
			tocU += timer() - ticU

			# Calculate V
			ticAlsqr = timer()
			V = self.alsqr_V(Utilde, lam, Rtilde, V)
			tocV += timer() - ticAlsqr

			# Find error
			Etilde = self.ptilde(Rtilde, Utilde, V)
			dl = abs(Etilde-E)/E; E = Etilde;
			if itr == 0: dl = Etilde;
			# self.vprint('alsqr: %d/%.3f/%.3f' % (itr,E,dl),'')
			itr += 1
		return Utilde, V, [tocV, itr, dl, E, tocU]

	def variance(self, P,R):
		# calculate variance for each (nym, item) pair
		Ridx = R.astype(bool)
		lam = P.dot(Ridx).toarray()
		lamInv = np.zeros_like(lam, dtype=float)
		lamInv[lam>0] = 1/lam[lam>0]

		A=(P.dot(R.multiply(R))).multiply(lamInv)
		B=(P.dot(R)).multiply(lamInv)
		B=B.multiply(B)
		var=A-B

		if self.varbound == True:
			var[var<self.sig] = self.sig
			var[var>10*self.sig] = 10*self.sig

		return var # returns sparse array, even though every element is filled.

	def inv_variance(self,P,R):
		# calculate reciprocal of variance for each (nym,item) pair
		Rvar_inv = self.variance(P,R)
		Rvar_inv[Rvar_inv < self.sig] = self.sig # avoid divide by zero
		np.reciprocal(Rvar_inv.data, out=Rvar_inv.data)
		return Rvar_inv.toarray() # return dense array

	def alsqr_U(self, V, lam, Rtilde, Utilde):
		# for regular MF the loop below is expensive since g equals #users and so big.  DL
		num_nyms = Rtilde.shape[0]
		Utilde = np.zeros((self.d, num_nyms), dtype=np.float32)
		for g in range(num_nyms):  # number of nyms
			if any(lam[g, :]):
				rt = Rtilde[g, :]
				if sp.isspmatrix(Rtilde):
					rt = rt.toarray().flatten()  # Rtilde is sparse, expand this row with real zeros

				Utilde[:, g] = linalg.solve(
					np.eye(self.d, dtype=float)  / self.sigU / self.sigU + V.dot(
						sp.diags(lam[g, :], 0).dot(V.T)), V.dot(rt * lam[g, :]))
		return Utilde

	def alsqr_V(self, Utilde, lam, Rtilde, outV):
		num_items = Rtilde.shape[1]
		V = np.zeros((self.d, num_items), dtype=np.float32)
		for v in range(num_items):
			if any(lam[:, v]):
					rt = Rtilde[:, v]
					if sp.isspmatrix(Rtilde):
						# Rtilde is sparse, expand this column to include real zeros
						# this might be expensive if there are many users
						rt = rt.toarray().flatten()

					if self.L2>0:
						# ridge regression
						V[:, v] = linalg.solve(np.eye(self.d, dtype=float)/self.sigV/self.sigV + Utilde.dot(sp.diags(lam[:, v], 0).dot(Utilde.T)), Utilde.dot(sp.diags(lam[:, v], 0).dot(rt)))
					else:
						# LASSO (least squares with L1 penalty)
						#clf = Lasso(alpha=0.001*g, fit_intercept=False, normalize=False, selection='random', tol=0.01)
						#clf.fit(sp.diags(lam[:, v], 0).dot(Utilde.T),sp.diags(lam[:, v], 0).dot(Rtilde[:, v]))
						clf = ElasticNet(alpha=self.alpha*g, fit_intercept=False, normalize=False, selection='random', tol=0.01)
						clf.fit(sp.diags(lam[:, v], 0).dot(Utilde.T),sp.diags(lam[:, v], 0).dot(Rtilde[:, v]))
						V[:, v] = clf.coef_

		return V


	def print_pred(self, P, Utilde, V, R):
		p = P.shape[0]
		R = R.tocsr() # essential, or nifty code below fails since tR.indices is row not column indices.  DL
		P = P.tocsr()
		cm_round = np.zeros((5,5) ,dtype=int)

		for i in range(p):
			tP = P[i, :]
			if tP.nnz == 0:
				continue

			tR = R[tP.indices, :]
			temp=Utilde[:, i].transpose().dot(V)[tR.indices]
			diff=tR.data-temp
			print(tR.data[diff>1]);print(temp[diff>1])			
		return 



	# Print analysis of nyms with number of users and average rating error
	def nym_errors(self, P, Utilde, V, R):
		p = P.shape[0]
		nym_err = np.zeros(p)
		nym_distrib = np.zeros(p, dtype=int)
		R = R.tocsr() # essential, or nifty code below fails since tR.indices is row not column indices.  DL
		P = P.tocsr()

		for i in range(p):
			tP = P[i, :]

			if tP.nnz == 0:
				nym_distrib[i] = 1
				continue

			# since tP is a csr_matrix tP.indices is the indices of the non-zero columns ie the users in nym i
			tR = R[tP.indices, :]
			nym_err[i] = np.square(tR.data - Utilde[:, i].transpose().dot(V)[tR.indices]).sum()
			nym_distrib[i] = tR.nnz

		return nym_err, nym_distrib

	def print_nym_analysis(self, P, Rnnz, nym_err, nym_distrib):
		if self.verbosity == 1:
			print("--- Nym distribution (MSE/rating): ", end='')

			Pcount = P.indptr[1:].copy(); Pcount[1:] -= Pcount[:-1].copy()

			# for i in range(P.shape[0]):
			# 	print("%d (%.3f), " % (Pcount[i], nym_err[i]/nym_distrib[i]), end='')

			print("Total error: %.3f." % (nym_err.sum()/Rnnz))

		return nym_err.sum()/Rnnz

	# Split nyms
	def split_nyms(self, idx, P, Utilde):
		# Start by removing under-used nyms
		if any(np.asarray(P.sum(1)).flatten() < self.smallest_nym):
			emptynyms = np.asarray(P.sum(1)).flatten() >= self.smallest_nym
			user_update = np.array(np.where(P[idx+np.invert(emptynyms), :].sum(0))[1]).flatten()

			P = P[emptynyms, :]
			Utilde = Utilde[:, emptynyms]
			idx = idx[emptynyms]
		else:
			user_update = np.array(np.where(P[idx, :].sum(0))[1]).flatten()

		idx = np.where(idx)[0]
		p = P.shape[0]

		# Create new P matrix with empty half
		newP = sp.lil_matrix((p+len(idx), P.shape[1]), dtype=int)
		newP[0:p, :] = P

		if newP.nnz == self.n:
			newP = newP.tocsc()

		# Create new Utilde
		newUtilde = np.zeros((self.d, p+len(idx)), dtype=np.float32)
		newUtilde[:, 0:p] = Utilde
		dist = np.zeros_like(idx)

		for i in range(0, len(idx)):
			# Find how isolated nym is
			if p == 1:
				dist[i] = 1
			else:
				temp = np.square(Utilde - Utilde[:, idx[i]].reshape(self.d, 1)).sum(0)
				dist[i] = temp[temp>0].min()

			newUtilde[:, p+i] = self.nym_split_noise*np.random.standard_normal(self.d)+Utilde[:, idx[i]]
		return newP, newUtilde, user_update, len(idx), dist

	def init_mtxs(self, P, R, Ridx):
		lam, Rtilde = self.recalc_lam_Rtilde(P, R, Ridx)
		Utilde = (np.random.randn(self.d, self.p0)).astype(np.float32)
		V = (np.random.randn(self.d, self.m)).astype(np.float32)

		return lam, Rtilde, Utilde, V

	def recalc_P(self, R, Utilde, V, users):
		tic = timer()
		i_values = np.zeros_like(users, dtype=np.float32)

		for i in range(len(users)):
			tR = R[users[i], :]

			# Find distance from user to each nym's ratings and return one of the mins at random
			soln = ( tR.data - Utilde.T.dot(V[:, tR.indices]) )**2 #  Calculate squared distance
			soln = soln.sum(1) # do a row-sum for each group

			i_values[i] = soln.argmin()

		return i_values, timer() - tic

	def apply_nym_updates(self, P, users, new_nyms):
		""" Update P matrix with users' new nyms """
		if P.nnz < self.n:
			P = P.tolil()  # ensure format is lil
			p = P.shape[0]

			old_nyms = sp.spmatrix.dot(np.array(range(1, p + 1)), P[:, users]) - 1  # Users' old nyms
			nym_updates = (new_nyms != user_nyms)  # Users whose nyms have changed

			P[old_nyms[nym_updates * (old_nyms > -1)], users[nym_updates * (old_nyms > -1)]] = 0  # Reset old nym
			P[new_nyms[nym_updates], users[nym_updates]] = 1  # Set new nym
		else:
			P = P.tocsc()  # ensure format is csc

			old_nyms = P[:, users].indices
			nym_updates = (new_nyms != old_nyms)

			P.indices[users[nym_updates]] = new_nyms[nym_updates]

		return P, nym_updates.sum()

	def vprint(self, str1, str0):
		if self.verbosity == 1:
			print(str1, end="")
		else:
			print(str0, end="")

		sys.stdout.flush()

		return None

	def cleanup(self):

		return None

	def messup(self):

		return None

	def nym_cm(self, P, Utilde, V, R):
		p = P.shape[0]
		R = R.tocsr() # essential, or nifty code below fails since tR.indices is row not column indices.  DL
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


	def run_BLC(self, ratings, verbose=1):
		# carry out BLC factorisation i.e. R = P^T Utilde^T V
		big_tic = timer()
		R = ratings['R']
		self.messup()
		self.verbosity = verbose

		R = R.tocsr() # make sure in row format so that Rcount etc are as expected. DL
		Ridx = R.astype(bool)
		Rcount = R.indptr[1:].copy(); Rcount[1:] -= Rcount[:-1].copy()
		self.n, self.m = R.shape
		p = self.p0

		# Initialise zero-matrix P
		if p == 1: P = sp.csr_matrix((1, self.n), dtype=int)
		else: P = sp.lil_matrix((p, self.n), dtype=int)
		# Initialise other variables
		lam, Rtilde, Utilde, V = self.init_mtxs(P, R, Ridx)

		err = float("inf"); tol_reached = False; new_nyms = 0.5; user_update = np.array(range(self.n))

		self.vprint("Itt/Users/Chng\t\tTimings (Rtilde, P, Alsqr V)\tIter\tDelta\n", "" )

		mainloop_time = timer()

		for split in range(self.max_iter_split):
			itt = 0
			split_time = timer()

			changes = 0
			while (itt == 0 or changes >= self.n*self.Ptol or tol_reached) and (itt < self.max_iter_p) \
					and (timer()-split_time < self.max_time_per_split):
				changes = 0; ittt = 0; tocP = 0.0; ticP = timer(); toc2 = 0.0;

				# Update P for each user at random
				tperm = np.random.permutation(user_update); perm = [];
				# Split updating into exponentially larger groups to aid convergence
				if itt == 0 and tol_reached == False:
					perm.append(tperm[0:min(len(user_update), int(self.log_frequency*2*new_nyms))])

					for i in range(2, 1+math.ceil(math.log(len(user_update)/2/new_nyms, self.log_frequency))):
						perm.append(tperm[int((self.log_frequency**(i-1))*2*new_nyms):min(len(user_update), int((self.log_frequency**i)*2*new_nyms))])
				else:
					perm = [np.array(tperm)]

				# Evaluate P for each user and run Alsqr in blocks
				for b in range(len(perm)):
					toc2 = 0.0
					if p > 1:
						users = perm[b]
						users = users[Rcount[users].argsort()[::-1]] # Sort users by number of ratings

						new_nyms, toc2 = self.recalc_P(R, Utilde, V, users)
						P, updated_nyms = self.apply_nym_updates(P, users, new_nyms)

						ittt += len(users)
						changes += updated_nyms
					else:
						# If only 1 nym, set subset of users
						P = P.tocsr()
						Pindices = np.append(P.indices, perm[b])
						Pindices = list(set(Pindices)) # remove duplicates. DL
						Pdata = np.ones((1, len(Pindices)), dtype=int).flatten()
						Pindptr = np.array([0, len(Pindices)], dtype=int)
						P = sp.csr_matrix((Pdata, Pindices, Pindptr), shape=(1, self.n), dtype=int)

						ittt += len(perm[b])
						changes += len(perm[b])

					tocP = timer() - ticP
					# Recalculate lam and Rtilde using new P before running alsqr
					P = P.tocsr()
					lam, Rtilde  = self.recalc_lam_Rtilde(P, R, Ridx)

					# Run alsqr, recasting P for optimised run-time
					Utilde, V, out = self.run_alsqr( Utilde, V, lam, Rtilde)

					self.vprint("%2d / %7d / %7d\t%7.3f / %7.3f / %7.3f\t%d\t%.2e\n"
								% (itt, ittt, changes, toc2, tocP, out[0], out[1], out[2]),"")

					ticP = timer()
					if timer() - split_time > self.max_time_per_split:
						print("TAKING TOO LONG TO SPLIT! ABORT (you might want to change max_time_per_split)")

				# Skip straight to nym splitting if only one nym
				if p == 1: break

				itt += 1

			# Analyse nyms
			print("run result")
			P = P.tocsr() # Best format for nym_analysis
			nym_err, nym_distrib = self.nym_errors(P, Utilde, V, R)
			err = self.print_nym_analysis(P, R.nnz, nym_err, nym_distrib)
			nym_mean_err = nym_err/nym_distrib

			######################
			print("run base line")
			P_base=sp.coo_matrix((np.ones(P.shape[1]), (np.zeros(P.shape[1]), range(P.shape[1]))), shape=P.shape).tocsr()
			nym_err_base, nym_distrib_base = self.nym_errors(P_base, Utilde, V, R)
			err_base = self.print_nym_analysis(P_base, R.nnz, nym_err_base, nym_distrib_base)
			##########################
			
			self.vprint("Main-loop time: %.3f\n" % (timer() - mainloop_time),"")
			mainloop_time = timer()

			# Check if we're done
			if err <= self.err_tol:
				if tol_reached == True:
					self.vprint("\nError tolerance of %.3f reached at %.3f, with %d nyms.\n" % (self.err_tol, err, p), "")
					break
				tol_reached = True
			elif p >= self.p1:
				if tol_reached == True:
					self.vprint("\nMaximum number of nyms of %d reached without reaching error tolerance (%.3f).\n"
								% (self.p1, self.err_tol), "")
					break
				tol_reached = True

			if tol_reached == True:
				self.vprint("\nDoubling stopped. Running last pass.\n","")
				user_update = np.array(range(self.n))
				new_nyms = p
				P = P.tocsc()
				continue

			# Split nyms
			P, Utilde, user_update, new_nyms, dist = self.split_nyms(nym_mean_err >= self.err_tol, P, Utilde)
			p = P.shape[0]
			self.vprint("\n--> Splitting %d nym(s) (distances: %s), increasing to %d. %d (%d%%) users affected. (Rtilde density: %.3f)\n"
						% (new_nyms, ", ".join(map(str, dist)), p, len(user_update), round(len(user_update)*100/self.n), (Rtilde > 1e-4).sum()/Rtilde.shape[0]/Rtilde.shape[1]),
						"")

		Rvar = self.variance(P,R)


		np.savetxt("Rvar.dat",Rvar.toarray())		
		self.vprint("Total time taken: %.3f\n\n" % (timer() - big_tic),"")
		self.cleanup()


		return Utilde, V, err, P

	def run_MF(self, ratings, verbose=1):
		# carry out regular matrix factorisation i.e R as U^T V

		big_tic = timer()
		self.messup()
		self.verbosity = verbose

		R = ratings['R']
		R = R.tocsr()  # make sure in row format so that Rcount etc are as expected. DL

		# Initialise U and V
		self.n, self.m = R.shape
		U = (np.random.randn(self.d, self.n)).astype(np.float32)
		V = (np.random.randn(self.d, self.m)).astype(np.float32)

		# Run alsqr
		lam = (R.astype(bool))*1
		lam = lam.toarray()
		#lam = Ridx.toarray()
		self.vprint("Starting MF ...\n", "")
		U, V, out = self.run_alsqr(U, V, lam, R)
		err = out[3]/R.nnz
		self.vprint("time: %7.3f U/%7.3f V\titers: %d\t MSE: %.2e\n" % (out[4], out[0], out[1], err),"")

		self.vprint("\nTotal time taken: %.3f\n\n" % (timer() - big_tic),"")

		self.cleanup()
		return U, V, err

	def validation(self, test, Utilde, V, P=None):
		test_R = test['R'].tocsr() # make sure R is in csr format
		p = Utilde.shape[1]
		n = test_R.shape[0]

		if P == None:
			# assign users to nyms based on the test_R ratings
			users = np.array(range(n))
			new_nyms, toc2 = self.recalc_P(test_R, Utilde, V, users)
			P=sp.csr_matrix((np.ones(n),(new_nyms,users)),shape=(p,n))
		else:
			P = P.tocsr()

		err = 0
		cm_round = np.zeros((3,3) ,dtype=int)
		for g in range(p):
			tP = P[g, :]
			test_idx = test_R.nonzero()
			if tP.nnz == 0:
				continue
			users_in_nym = tP.indices
			if users_in_nym.size == 0:
				continue
			test_users_in_nym = np.in1d(test_idx[0], users_in_nym) #boolean mask for test_idx[0]
			test_columns = test_idx[1][test_users_in_nym]
			pred = Utilde[:,g].transpose().dot(V[:,test_columns])
			if self.clip:
				pred = np.clip(pred,self.clip_min,self.clip_max)
			if self.round:
				pred = np.rint(pred)
			err += np.square(test_R.data[test_users_in_nym]-pred).sum()

			# confusion matrix
			temp1=test_R.data[test_users_in_nym]
			temp2=np.round(pred)
			temp2[temp2<1]=1; 
			temp2[temp2>3]=3
			for i in range(temp2.shape[0]):
				c=int(temp2[i])-1 # pridicted label
				r=int(temp1[i])-1 # true label
				cm_round[r,c]+=1

		return np.sqrt(err/test_R.nnz),cm_round

	def naive(self, ratings, test):
		R = ratings['R']
		test_R = test['R']

		total_mean = R.data.mean()
		nrows,ncolumns = R.shape
		sums = R.sum(axis=1).A1
		counts = np.diff(R.indptr)
		averages_rows = sums / counts
		bias_rows = averages_rows - total_mean
		R2 = sp.csr_matrix(R.todense().transpose())
		sums = R2.sum(axis=1).A1
		counts = np.diff(R2.indptr)
		averages_columns = sums / counts
		bias_columns = averages_columns - total_mean

		bias_rows_matrix = np.tile(bias_rows,(ncolumns,1)).transpose()
		bias_columns_matrix = np.tile(bias_columns,(nrows,1))
		final_pred = bias_rows_matrix + bias_columns_matrix + total_mean
		print(final_pred.shape)
		print(R.shape)
		print(test_R.shape)

		err = np.square(np.array(final_pred[test_R.nonzero()[0],test_R.nonzero()[1]]).flatten() - test_R.data).sum()

		pred_columns = R.mean(axis=1)
		prediction = pred_columns[test_R.nonzero()[1]]
		n = test_R.nnz
		err2 = np.square(test.data - prediction).sum()
		print(np.sqrt(err2/n))
		return np.sqrt(err/n)


	def split(self,ratings,test_ratio,seed=None,thresh=3,verbose=1):
		self.vprint("Splitting dataset with test_ratio=%s\n"%test_ratio,'')

		R = ratings['R']
		R = R.tocsr() # make sure R is in csr format

		shap = R.shape
		R_users = R.nonzero()[0]
		R_items = R.nonzero()[1]

		np.random.seed(seed)
		test_idx = np.random.choice(range(R.nnz),int(R.nnz*test_ratio),replace=False) # sample without replacement
		#test_idx = list(set(test_idx)) # remove duplicates. DL
		test_R = sp.coo_matrix((R.data[test_idx], (R_users[test_idx], R_items[test_idx])),shape=shap)
		test_R = test_R.tocsr()
		test_R.sort_indices()

		train_data = np.delete(R.data, test_idx)
		train_users = np.delete(R_users, test_idx)
		train_items = np.delete(R_items, test_idx)
		train_R = sp.coo_matrix((train_data, (train_users, train_items)),shape=shap)
		train_R = train_R.tocsr()
		train_R.sort_indices()

		# remove users from test data for which there is insufficient training data (and so P entry for that user
		# is not determined from training data)
		entries_per_row = train_R.getnnz(axis=1)
		too_few = np.array(range(shap[0]))[entries_per_row<thresh]
		for i in too_few:
			test_R.data[test_R.indptr[i]:test_R.indptr[i + 1]] = 0
		test_R.eliminate_zeros()

		self.vprint("Removed %d rows from test_R due to too few training samples.\n"%too_few.size,"")

		#print("CONSISTENCY TEST ALEX")
		#errfloat = (R-(train+test)).mean()
		#if errfloat < 0.01:
		#	print("PASSED!")
		#else:
		#	print("FAILED!")
		train = {}
		train['R'] = train_R

		test = {}
		test['R'] = test_R

		return train,test

class BLC_CPU(BLC):
	def __init__(self):
		BLC.__init__(self)
		self.num_cores = multiprocessing.cpu_count() - 1
		self.alsqr_U_parallel = alsqr_U_parallel
		self.alsqr_V_parallel = alsqr_V_parallel
		# self.nym_errors_parallel = nym_errors_parallel
		self.recalc_P_parallel = recalc_P_parallel

	def recalc_P(self, R, Utilde, V, users):
		tR = R[users, :]
		tic = timer()
		recalc_P_partial = partial(self.recalc_P_parallel, R=tR, Utilde=Utilde, V=V)
		i_values = self.pool.map(recalc_P_partial, range(len(users)))

		return np.array(i_values), timer() - tic

	def alsqr_U(self, V, lam, Rtilde, outU):
		alsqr_U_partial = partial(self.alsqr_U_parallel, V, lam, Rtilde, outU,
								  1 / self.sigU / self.sigU)
		num_nyms = Rtilde.shape[0]
		Uvals = self.pool.map(alsqr_U_partial, range(num_nyms))
		U = np.zeros((self.d, num_nyms), dtype=np.float32)

		for v in range(num_nyms):
			if any(Uvals[v] != 0):
				U[:, v] = Uvals[v]
		return U

	def alsqr_V(self, Utilde, lam, Rtilde, outV):
		alsqr_V_partial = partial(self.alsqr_V_parallel, Utilde, lam, Rtilde, outV, 1/self.sigV/self.sigV, self.L2, self.alpha)
		Vvals = self.pool.map(alsqr_V_partial, range(self.m))
		V = np.zeros((self.d, self.m), dtype=np.float32)

		for v in range(self.m):
			if any(Vvals[v] != 0):
				V[:, v] = Vvals[v]

		return V

	def nym_errors(self, P, Utilde, V, R):
		nym_errors_partial = partial(nym_errors_parallel, P, R, Utilde, V)
		nym_err = self.pool.map(nym_errors_partial, range(P.shape[0]))

		nym_err = np.array(nym_err)
		nym_distrib = nym_err[:, 1]
		nym_err = nym_err[:, 0]

		return nym_err, nym_distrib

	def cleanup(self):
		self.pool.terminate()
		return None

	def messup(self):
		self.pool = multiprocessing.Pool(processes=self.num_cores)
		return None


class BLC_GPU(BLC_CPU):
	def __init__(self):
		BLC_CPU.__init__(self)
		self.max_GPU_block = 100000	# Most users in a single GPU execution

	def messup(self):
		BLC_CPU.messup(self)

		self.ctx = cl.create_some_context(interactive=False)
		self.queue = cl.CommandQueue(self.ctx)
		self.prg = cl.Program(self.ctx, self.cl_recalc_P()).build()
		self.P_GPU = self.prg.recalc_P

		return None


	def cl_recalc_P(self):
		str = """
		__kernel void recalc_P( __global const int *mv,
				__global const float *rt,
				__global const int *Uidx,
				__global const float *U,
				__global const float *V,
				__global const int *dims,
				__global float *out)
		{
		int gid = get_global_id(0);

		int d = dims[0]; // Number of feature dimensions
		int p = dims[1]; // Number of nyms

		int u = gid/p;
		int i = gid % p;

		int z;
		float row_sum = 0.0;
		float entry;

		for (z = Uidx[u]; z < Uidx[u+1]; z++) // Loop through users' ratings
		{
			entry = rt[z];
		"""

		for i in range(self.d):
			str += "		entry -= U[i * d + {}] * V[mv[z] * d + {}];\n".format(i, i)

		str += """
			row_sum += entry*entry;
		}

		out[gid] = row_sum;
		return;
		}
		"""
		return str

	def recalc_P(self, R, Utilde, V, users):
		toc = 0.0
		p = Utilde.shape[1]
		i_values = np.zeros(p*users.shape[0], dtype=np.float32)

		# Set-up GPU parameters
		Utildebar = Utilde.T.flatten().astype(np.float32)
		Vbar = V.T.flatten().astype(np.float32)
		dims = np.array([self.d, p, self.m], np.int32) # dims: d, p, m

		ut_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Utildebar)
		v_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Vbar)
		dim_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dims)

		for i in range(0, math.ceil(len(users)/self.max_GPU_block)):
			temp_users = users[self.max_GPU_block*i:min(len(users), self.max_GPU_block*(i+1))]
			temp_i_values = np.zeros(p*temp_users.shape[0]).astype(np.float32)

			# Get user indices, movies and ratings for GPU
			tR = R[temp_users, :]
			GPU_Uidx = tR.indptr.astype(np.int32)
			GPU_mv = tR.indices.astype(np.int32)
			GPU_rt = tR.data.astype(np.float32)

			mv_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=GPU_mv)
			rt_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=GPU_rt)
			uidx_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=GPU_Uidx)
			dest_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, temp_i_values.nbytes)

			tic = timer()
			Pcalc = self.P_GPU(self.queue, temp_i_values.shape, None, mv_buf, rt_buf, uidx_buf, ut_buf, v_buf, dim_buf, dest_buf)
			Pcopy = cl.enqueue_copy(self.queue, temp_i_values, dest_buf, wait_for=[Pcalc])
			Pcopy.wait()
			toc += timer() - tic

			i_values[self.max_GPU_block*i*p:min(users.shape[0]*p, self.max_GPU_block*p*(i+1))] = temp_i_values

		i_values = i_values.reshape(users.shape[0], p).argmin(1) # Find min error for each user

		return i_values, toc

class BLC_GPU_float(BLC_GPU):
	def __init__(self):
		print("ERROR: BLC_GPU_float() is a deprecated class. Use BLC_GPU instead.\nAborting...")
		exit()

# Initialise R user ratings matrix
def init_R(p, n, m, d, sparsity, noise=None, Vsparse=-1, verbose=1):
	P = sp.lil_matrix((p, n), dtype=int)
	for u in range(0, n):
		P[np.random.randint(p), u] = 1
	P = P.tocsr()
	Pcount = P.indptr[1:].copy(); Pcount[1:] -= Pcount[:-1].copy()
	if verbose: print("Nym distribution: %s" % ", ".join(map(str, Pcount)))

	Utilde = np.zeros((d, p), dtype=float)
	Utilde[:, 0] = np.random.randn(d)

	for i in range(1, p):
		candidate = np.random.randn(d, 1000)
		min_vals = np.zeros(candidate.shape[1])

		for j in range(0, candidate.shape[1]):
			min_vals[j] = np.square(Utilde[:, 0:i] - candidate[:, j].reshape(d, 1)).sum(0).min()

		Utilde[:, i] = candidate[:, np.argmax(min_vals)]
	V = np.random.randn(d, m)
	if (Vsparse>0):
		temp = np.random.permutation(d*m)[0:round(Vsparse*d*m)]
		V[temp//m, temp%m] = 0

	# R = (P.transpose().dot(Utilde.transpose()) + np.random.randn(n, d)/100).dot(V)
	#R = P.transpose().dot(Utilde.transpose()).dot(V)
	# R = np.random.random_integers(1, 10, (n, m))
	#temp = np.random.permutation(n*m)[0:round(sparsity*n*m)]
	#R[temp//m, temp%m] = 0

	Rtilde = Utilde.transpose().dot(V)
	R = P.transpose().dot(Rtilde)

	if noise is not None:
		R = R + noise

	temp = np.random.permutation(n*m)[0:round(sparsity*n*m)]
	R[temp//m, temp%m] = 0
	return sp.csr_matrix(R)

def remove_mean(R):
  # remove column means from sparse matrix (need to be careful not to touch zero entries in matrix). DL
  # note: R must be a float matrix, not integer
  # To do: removing mean may cause some entries to become zero and so be mistaken as missing entries (the sparse
  # matrix representation treats the two as being the same).  Could add check for this, but haven't bothered just
  # now as seems likely to be rare
	X=sp.csc_matrix(R)
	rows, cols = X.shape
	col_start_stop = as_strided(X.indptr, shape=(rows, 2),
                            strides=2*X.indptr.strides)
	for col, (start, stop) in enumerate(col_start_stop):
		data = X.data[start:stop]
		m = data.sum()/(data!=0).sum()
		print('%d %f %f %f' %(col,m,data.sum(),(data!=0).sum()))
		print(data)
		data -= m
		print(data)
	return sp.csr_matrix(X)

def prepare_R(ratings, verbose=1):
	R = ratings['R']

	columns = np.asarray(R.sum(0)>0).flatten()
	if (R.sum(0)==0).sum() > 0:
		if verbose: print("Removing columns...", end="")
		R = R.tocsc()
		columns = np.asarray(R.sum(0)>0).flatten()
		R = R[:, columns]

	# Remove empty rows (users)
	R = sp.csr_matrix(R) # Convert to sparse row matrix for remainder of programme
	rows = np.asarray(R.sum(1)>0).flatten()
	if (R.sum(1)==0).sum() > 0:
		if verbose: print("Removing rows...", end="")
		R = R[rows,:]

	R.eliminate_zeros()
	R.sort_indices()
	ratings['R'] = R
	return ratings,np.where(rows),np.where(columns)

# Parallelisation functions required to be outside class for multiprocessing
def alsqr_V_parallel(Utilde, lam, Rtilde, outV, sig2, L2, alpha, v):
	if any(lam[:, v]):
		if L2>0:
			rt = Rtilde[:, v]
			if sp.isspmatrix(Rtilde):
				rt = rt.toarray().flatten()  # Rtilde is sparse, expand this column with real zeros
			return linalg.solve(np.eye(Utilde.shape[0], dtype=np.float32)*sig2 + Utilde.dot(sp.diags(lam[:, v], 0).dot(Utilde.T)), Utilde.dot(sp.diags(lam[:, v], 0).dot(rt)))
		else:
			clf = ElasticNet(alpha=alpha*g, fit_intercept=False, normalize=False, selection='random', tol=0.01)
			clf.fit(sp.diags(lam[:, v], 0).dot(Utilde.T),sp.diags(lam[:, v], 0).dot(Rtilde[:, v]))
			return clf.coef_

	return np.zeros(Utilde.shape[0])

def alsqr_U_parallel(V, lam, Rtilde, outU, sig2, g):
	d = V.shape[0]
	if any(lam[g, :]):
		rt = Rtilde[g, :]
		if sp.isspmatrix(Rtilde):
			rt = rt.toarray().flatten()  # Rtilde is sparse, expand this row with real zeros
		return linalg.solve(np.eye(d, dtype=float) *sig2 + V.dot(sp.diags(lam[g, :], 0).dot(V.T)), V.dot(rt * lam[g, :]))
	return np.zeros(V.shape[0])

def recalc_P_parallel(u, R, Utilde, V):
	tR = R[u, :]

	# Find distance from user to each nym's ratings and return one of the mins at random
	soln = (tR.data - Utilde.T.dot(V[:, tR.indices]))**2 #  Calculate square error
	soln = soln.sum(1) # do a row-sum for

	return soln.argmin()

def nym_errors_parallel(P, R, Utilde, V, p):
	tP = P[p, :]
	if tP.nnz == 0:
		return [0, 1] # Return 0.1 to avoid zero division
	tR = R[tP.indices, :]
	return [np.square(tR.data - Utilde[:, p].transpose().dot(V)[tR.indices]).sum(), tR.nnz]

