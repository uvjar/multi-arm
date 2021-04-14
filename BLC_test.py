#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic testing for BLC()
"""
import unittest
import nose
import BLC
import numpy as np
import scipy.sparse as sp
import numpy.testing as npt

class Test_standalone(unittest.TestCase):
	# these are all tests of standalone routines

	def test_prepare_R_scalar(self):
		# prep of 1x1 matrix
		ratings = {}
		ratings['R'] = sp.coo_matrix(([1], ([0], [0])), dtype=np.float32)
		ratings2, original_rows, original_columns = BLC.prepare_R(ratings, verbose=0)
		self.assertEqual(original_rows, ([0],))
		self.assertEqual(original_columns, ([0],))
		self.assertEqual(ratings, ratings2)

	def test_prepare_R_emptyrow(self):
		# removal of a row
		ratings = {}
		ratings['R'] = sp.coo_matrix(([0, 1], ([0, 1], [0, 0])), dtype=np.float32)
		ratings2, original_rows, original_columns = BLC.prepare_R(ratings, verbose=0)
		self.assertEqual(original_rows, ([1],))
		self.assertEqual(original_columns, ([0],))
		self.assertEqual(ratings2['R'].shape, (1, 1))
		diff = (ratings2['R'] != sp.coo_matrix(([1], ([0], [0])), dtype=np.float32).tocsr())
		self.assertEqual(diff.nnz, 0)

	def test_prepare_R_emptycol(self):
		# removal of a column
		ratings = {}
		ratings['R'] = sp.coo_matrix(([0, 1], ([0, 0], [0, 1])), dtype=np.float32)
		ratings2, original_rows, original_columns = BLC.prepare_R(ratings, verbose=0)
		self.assertEqual(original_rows, ([0],))
		self.assertEqual(original_columns, ([1],))
		self.assertEqual(ratings2['R'].shape, (1, 1))
		diff = (ratings2['R'] != sp.coo_matrix(([1], ([0], [0])), dtype=np.float32).tocsr())
		self.assertEqual(diff.nnz, 0)

	def test_split_R_as_csc(self):
		# two users, two items. R in csc format -- this caused a silent error previously
		B = BLC.BLC()
		R=sp.coo_matrix(([1,2,3,4], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsc()
		ratings={}; ratings['R']=R
		train, test = B.split(ratings,0.5,verbose=0)
		#print(test['R']); print("train:"); print(train['R']);print("sum:")
		#print(test['R']+train['R'])
		#print("-")
		#print(R)
		self.assertTrue((((test['R']+train['R']).tocsr()-R)**2).sum()<0.001)
		ratings['R']=ratings['R'].tocsr()
		train, test = B.split(ratings,0.5,verbose=0)
		self.assertTrue((((test['R']+train['R']).tocsr()-R)**2).sum()<0.001)

	def test_validation_one(self):
		B = BLC.BLC();
		R=[[1]]
		Utilde=np.array([[1]])
		V=np.array([[1]])
		P=sp.csr_matrix([[1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e,0.0)

	def test_validation_one_csc(self):
		B = BLC.BLC();
		R=[[1]]
		Utilde=np.array([[1]])
		V=np.array([[1]])
		P=sp.csc_matrix([[1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e,0.0)

	def test_validation_one_by_two(self):
		B = BLC.BLC();
		R=[[1,2]]
		Utilde=np.array([[1]])
		V=np.array([[1,2]])
		P=sp.csr_matrix([[1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e, 0.0)

	def test_validation_two_by_one_p1(self):
		B = BLC.BLC();
		R=[[1],[1]]
		Utilde=np.array([[1]])
		V=np.array([[1]])
		P=sp.csr_matrix([[1,1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e, 0.0)

	def test_validation_two_by_one_p2(self):
		B = BLC.BLC();
		R=[[1],[2]]
		Utilde=np.array([[1,2]])
		V=np.array([[1]])
		P=sp.csr_matrix([[1,0],[0,1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e, 0.0)

	def test_validation_two_by_two_p2(self):
		B = BLC.BLC();
		R=[[1,2],[2,4]]
		Utilde=np.array([[1,2]])
		V=np.array([[1,2]])
		P=sp.csr_matrix([[1,0],[0,1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e, 0.0)

	def test_validation_two_by_two_d2(self):
		B = BLC.BLC();
		R=[[4,6],[4,6]]
		Utilde=np.array([[1],[1]])
		V=np.array([[1,2],[3,4]])
		P=sp.csr_matrix([[1,1]])
		test={}; test['R']=sp.csr_matrix(R)
		e = B.validation(test, P, Utilde, V)
		self.assertEqual(e, 0.0)

class Test_BLCpieces(unittest.TestCase):
	# these are tests of worker routines used by BLC -- these should be tested before run_BLC() is tested

	def test_ptilde_one(self):
		B = BLC.BLC()
		Utilde=np.array([[1]]); V=np.array([[1]]); Rtilde=np.array([[1]])
		e = B.ptilde(Rtilde, Utilde, V)
		self.assertEqual(e, 0)

	def test_ptilde_two(self):
		B = BLC.BLC()
		Utilde=np.array([[1]]); V=np.array([[1]]); Rtilde=np.array([[3]])
		e = B.ptilde(Rtilde, Utilde, V)
		self.assertEqual(e, 4)

	def test_ptilde_three(self):
		B = BLC.BLC()
		Utilde=np.array([[2,1]]); V=np.array([[1]]); Rtilde=np.array([[1],[1]])
		e = B.ptilde(Rtilde, Utilde, V)
		self.assertEqual(e, 1)

	def test_ptilde_four(self):
		B = BLC.BLC()
		Utilde=np.array([[1,2]]); V=np.array([[1,1]]); Rtilde=np.array([[1,1],[2,3]])
		e = B.ptilde(Rtilde, Utilde, V)
		self.assertEqual(e, 1)

	def test_ptilde_four_csr(self):
		B = BLC.BLC()
		Utilde=np.array([[2,1]]); V=np.array([[1]]); Rtilde=sp.csr_matrix(np.array([[1],[1]]))
		e = B.ptilde(Rtilde, Utilde, V)
		self.assertEqual(e, 1)

	def test_nym_errors_scalar(self):
		# one user, one item, one nym, one rating !
		B = BLC.BLC()
		P=sp.coo_matrix(([1], ([0], [0])))
		P = P.tocsr()
		Utilde=np.array([[1]])
		V=np.array([[1]])
		R=sp.coo_matrix(([1], ([0], [0])))
		R=R.tocsr()
		nym_err,nym_distrib = B.nym_errors(P, Utilde, V, R)
		self.assertEqual(nym_err,0)
		self.assertEqual(nym_distrib,[1])

	def test_nym_errors_oneuser(self):
		# one user, two items
		B = BLC.BLC()
		P=sp.coo_matrix(([1], ([0], [0])))
		P = P.tocsr()
		Utilde=np.array([[1]])
		V=np.array([[1,2]]) # row vector
		R=sp.coo_matrix(([1,2], ([0,0], [0,1])))
		R=R.tocsr()
		nym_err,nym_distrib = B.nym_errors(P, Utilde, V, R)
		self.assertEqual(nym_err,0)
		self.assertEqual(nym_distrib,[2])

	def test_nym_errors_twousers_diffnyms(self):
		# two users, two items
		B = BLC.BLC()
		P=sp.coo_matrix(([1,1], ([0,1], [0,1])))
		P = P.tocsr()
		Utilde=np.array([[1,2]]) # row vector
		V=np.array([[1,2]]) # row vector
		R=sp.coo_matrix(([1,2,2,4], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsr()
		nym_err,nym_distrib = B.nym_errors(P, Utilde, V, R)
		npt.assert_array_equal(nym_err,[0,0])
		npt.assert_array_equal(nym_distrib,[2,2])

	def test_nym_errors_twousers_samenym(self):
		# two users, two items
		B = BLC.BLC()
		P=sp.coo_matrix(([1,1], ([0,0], [0,1])))
		P = P.tocsr()
		Utilde=np.array([[1]])
		V=np.array([[1,2]]) # row vector
		R=sp.coo_matrix(([1,2,1,2], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsr()
		nym_err,nym_distrib = B.nym_errors(P, Utilde, V, R)
		npt.assert_array_equal(nym_err,[0])
		npt.assert_array_equal(nym_distrib,[4])

	def test_nym_errors_R_as_csc(self):
		# two users, two items. R and P in csc format -- either caused a silent error previously
		B = BLC.BLC()
		P=sp.coo_matrix(([1,1], ([0,0], [0,1])))
		P = P.tocsc()
		Utilde=np.array([[1]])
		V=np.array([[1,2]]) # row vector
		R=sp.coo_matrix(([1,2,1,2], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsc()
		nym_err,nym_distrib = B.nym_errors(P, Utilde, V, R)
		npt.assert_array_equal(nym_err,[0])
		npt.assert_array_equal(nym_distrib,[4])

	def test_recalc_lam_Rtilde_p1(self):
		B = BLC.BLC()
		P = sp.csr_matrix([[1,1]])
		R = sp.csr_matrix([[1,2],[1,2]])
		Ridx = R.astype(bool)
		lam, Rtilde = B.recalc_lam_Rtilde(P, R, Ridx)
		npt.assert_array_equal(lam,[[2,2]])
		npt.assert_array_equal(Rtilde,[[1,2]])

	def test_recalc_lam_Rtilde_p2(self):
		B = BLC.BLC()
		P = sp.csr_matrix([[1,0],[0,1]])
		R = sp.csr_matrix([[1,2],[1,0]])
		Ridx = R.astype(bool)
		lam, Rtilde = B.recalc_lam_Rtilde(P, R, Ridx)
		npt.assert_array_equal(lam,[[1,1],[1,0]])
		npt.assert_array_equal(Rtilde,[[1,2],[1,0]])

	def test_alsqr_V_one_csr(self):
		B = BLC.BLC()
		Rtilde = sp.csr_matrix([[1,2]])
		Utilde = np.array([[1]])
		lam = np.array([[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		V = B.alsqr_V(Utilde, lam, Rtilde,[0])
		npt.assert_array_almost_equal(V,[[1,2]],2)

	def test_alsqr_V_two_csr(self):
		B = BLC.BLC()
		Rtilde = sp.csr_matrix([[1,2]])
		Utilde = np.array([[2]])
		lam = np.array([[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		V = B.alsqr_V(Utilde, lam, Rtilde, [0])
		npt.assert_array_almost_equal(V,[[0.5,1]],2)

	def test_alsqr_V_three_csr(self):
		B = BLC.BLC()
		Rtilde = sp.csr_matrix([[1,2],[2,4]])
		Utilde = np.array([[1,2]])
		lam = np.array([[1,1],[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		V = B.alsqr_V(Utilde, lam, Rtilde, [0])
		npt.assert_array_almost_equal(V,[[1,2]],2)

	def test_alsqr_V_three(self):
		B = BLC.BLC()
		Rtilde = np.array([[1,2],[2,4]])
		Utilde = np.array([[1,2]])
		lam = np.array([[1,1],[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		V = B.alsqr_V(Utilde, lam, Rtilde, [0])
		npt.assert_array_almost_equal(V,[[1,2]],2)

	def test_alsqr_U_one_csr(self):
		B = BLC.BLC()
		Rtilde = sp.csr_matrix([[1,2]])
		V = np.array([[1,2]])
		lam = np.array([[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		Utilde = B.alsqr_U(V, lam, Rtilde, [0])
		npt.assert_array_almost_equal(Utilde,[[1]],2)

	def test_alsqr_U_two(self):
		B = BLC.BLC()
		Rtilde = np.array([[1,2],[2,4]])
		V = np.array([[1,2]])
		lam = np.array([[1,1],[1,1]])
		B.m = Rtilde.shape[1]; B.d = 1
		Utilde = B.alsqr_U(V, lam, Rtilde, [0])
		#npt.assert_array_almost_equal(Utilde,[[1,2]],2)

	def test_alsqr_U_three(self):
		B = BLC.BLC()
		Rtilde = np.array([[1,2],[3,0]])
		V = np.array([[1,2]])
		lam = np.array([[1,1],[1,0]])
		B.m = Rtilde.shape[1]; B.d = 1
		Utilde = B.alsqr_U(V, lam, Rtilde, [0])
		npt.assert_array_almost_equal(Utilde,[[1,3]],2)

	def test_alsqr_U_three_MP(self):
		B = BLC.BLC_CPU()  # multiprocessing version
		B.messup()
		Rtilde = np.array([[1,2],[3,0]])
		V = np.array([[1,2]])
		lam = np.array([[1,1],[1,0]])
		B.m = Rtilde.shape[1]; B.d = 1
		Utilde = B.alsqr_U(V, lam, Rtilde, [0])
		npt.assert_array_almost_equal(Utilde,[[1,3]],2)

	def test_variance(self):
		B = BLC.BLC()
		P = sp.csr_matrix([[1, 0, 1], [0, 1, 0]])
		R = sp.csr_matrix([[1, 2], [3, 5], [5, 1]])
		B.varbound = False
		var = B.variance(P, R)
		npt.assert_array_equal(var.toarray(), [[4., 0.25], [0., 0.]])  # matrix calculation OK
		B.varbound = True
		var = B.variance(P, R)
		npt.assert_array_equal(var.toarray(), [[min(4,10*B.sig), max(0.25,B.sig)], [B.sig, B.sig]])  # matrix calculation OK

	def test_inv_variance(self):
		B = BLC.BLC()
		P = sp.csr_matrix([[1, 0, 1], [0, 1, 0]])
		R = sp.csr_matrix([[-1, 2], [3, 0.5], [0.25, 1]])
		B.varbound = False
		ivar = B.inv_variance(P, R)
		npt.assert_array_equal(ivar, [[2.56, 4], [1/B.sig, 1/B.sig]])

class TestBLC(unittest.TestCase):
	# tests of run_BLC and run_MF

	def test_run_MF_one_by_two_d5(self):
		B = BLC.BLC();
		B.d = 5
		R=sp.coo_matrix(([1,2], ([0,0], [0,1])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U_MF, V_MF, err_MF = B.run_MF(ratings,verbose=0)
		self.assertEqual(U_MF.shape,(5,1))
		self.assertEqual(V_MF.shape,(5,2))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_MF_two_by_one_d2(self):
		B = BLC.BLC();
		B.d = 2
		R=sp.coo_matrix(([1,2], ([0,1], [0,0])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U_MF, V_MF, err_MF = B.run_MF(ratings,verbose=0)
		self.assertEqual(U_MF.shape,(2,2))
		self.assertEqual(V_MF.shape,(2,1))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_MF_two_by_two_d10(self):
		B = BLC.BLC();
		B.d=10
		R=sp.coo_matrix(([1,2,1,2], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U_MF, V_MF, err_MF = B.run_MF(ratings,verbose=0)
		self.assertEqual(U_MF.shape,(10,2))
		self.assertEqual(V_MF.shape,(10,2))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_MF_ten_by_ten_rand_d1(self):
		B = BLC.BLC();
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1
		U_MF, V_MF, err_MF = B.run_MF(ratings, verbose=0)
		self.assertEqual(U_MF.shape,(1,10))
		self.assertEqual(V_MF.shape,(1,10))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.5)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_MF_ten_by_ten_rand_d1_MP(self):
		B = BLC.BLC_CPU(); # multiprocessing version
		B.messup()
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1;
		U_MF, V_MF, err_MF = B.run_MF(ratings, verbose=0)
		self.assertEqual(U_MF.shape,(1,10))
		self.assertEqual(V_MF.shape,(1,10))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.5)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_MF_ten_by_ten_rand_d1_GPU(self):
		B = BLC.BLC_GPU(); # multiprocessing and GPU version
		B.messup()
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1; B.max_iter_alsqr=10
		U_MF, V_MF, err_MF = B.run_MF(ratings, verbose=0)
		self.assertEqual(U_MF.shape,(1,10))
		self.assertEqual(V_MF.shape,(1,10))
		err = ((U_MF.transpose().dot(V_MF) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.5)
		self.assertTrue(abs(err-err_MF)<0.01)

	def test_run_BLC_one_by_two_d5(self):
		B = BLC.BLC(); B.verbosity=0
		B.d = 5; B.p = 1
		R=sp.coo_matrix(([1,2], ([0,0], [0,1])))
		R=R.tocsr()
		ratings = {}; ratings['R'] = R
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,1))
		self.assertEqual(U.shape,(5,P.shape[0]))
		self.assertEqual(V.shape,(5,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_one_by_two_d5_var(self):
		B = BLC.BLC(); B.verbosity=0
		B.d = 5; B.p = 1
		R=sp.coo_matrix(([1,2], ([0,0], [0,1])))
		R=R.tocsr()
		ratings = {}; ratings['R'] = R
		B.walsqr = True
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,1))
		self.assertEqual(U.shape,(5,P.shape[0]))
		self.assertEqual(V.shape,(5,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_var(self):
		B = BLC.BLC(); B.verbosity=0
		B.d = 1; B.p1 = 2; B.sig=0.01
		R=sp.csr_matrix([[-20,0.12],[20,0.121],[100,0.11]]) # first item has high variance, second item has zero variance
		R=R.tocsr()
		ratings = {}; ratings['R'] = R
		B.walsqr = True; B.varbound=False
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(2,3))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		Rhat1=P.transpose().dot(U.transpose().dot(V))
		#print(Rhat1)
		self.assertTrue(abs(err-err2)<0.01)
		B.walsqr = False
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		Rhat=P.transpose().dot(U.transpose().dot(V))
		self.assertTrue(abs(Rhat[0,1]-R[0,1])>abs(Rhat1[0,1]-R[0,1])) # error on zero variance item is higher without weighting

	def test_run_BLC_two_by_one_d2(self):
		B = BLC.BLC();
		B.d = 2
		R=sp.coo_matrix(([1,2], ([0,1], [0,0])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U, V, err2,P = B.run_BLC(ratings,verbose=0)
		self.assertEqual(P.shape,(2,2))
		self.assertEqual(U.shape,(2,P.shape[0]))
		self.assertEqual(V.shape,(2,1))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_two_by_two_d10_1nym(self):
		B = BLC.BLC();
		B.d=10
		# R matrix has identical rows, so 1 nym enough
		R=sp.coo_matrix(([1,2,1,2], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U, V, err2,P = B.run_BLC(ratings,verbose=0)
		self.assertEqual(P.shape, (1,2))
		self.assertEqual(U.shape,(10,P.shape[0]))
		self.assertEqual(V.shape,(10,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_two_by_two_d10_2nyms(self):
		B = BLC.BLC();
		B.d=10
		# change 2nd row of R, so now should need 2 nyms rather than 1
		R=sp.coo_matrix(([1,2,1,3], ([0,0,1,1], [0,1,0,1])))
		R=R.tocsc()
		ratings = {}; ratings['R'] = R
		U, V, err2,P = B.run_BLC(ratings,verbose=0)
		self.assertEqual(P.shape, (2,2))
		self.assertEqual(U.shape,(10,P.shape[0]))
		self.assertEqual(V.shape,(10,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_ten_by_ten_rand_d1_p1(self):
		B = BLC.BLC();
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1; B.p1=1
		U,V,err2,P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,10))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,10))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_ten_by_ten_rand_d1_p1_var(self):
		B = BLC.BLC();
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1; B.p1=1
		B.walsqr = True
		U,V,err2,P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,10))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,10))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_ten_by_ten_rand_d1_p1_MP(self):
		B = BLC.BLC_CPU();  # multiprocessing version
		B.messup()
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1; B.p1=1
		U,V,err2,P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,10))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,10))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

	def test_run_BLC_var_MP(self):
		B = BLC.BLC_CPU(); B.verbosity=0 # multiprocessing version
		B.d = 1; B.p1 = 2; B.sig=0.01
		R=sp.csr_matrix([[-20,0.12],[20,0.121],[100,0.11]]) # first item has high variance, second item has zero variance
		R=R.tocsr()
		ratings = {}; ratings['R'] = R
		B.walsqr = True; B.varbound=False
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(2,3))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,2))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		Rhat1=P.transpose().dot(U.transpose().dot(V))
		#print(Rhat1)
		self.assertTrue(abs(err-err2)<0.01)
		B.walsqr = False
		U, V, err2, P = B.run_BLC(ratings, verbose=0)
		Rhat=P.transpose().dot(U.transpose().dot(V))
		self.assertTrue(abs(Rhat[0,1]-R[0,1])>abs(Rhat1[0,1]-R[0,1])) # error on zero variance item is higher without weighting

	def test_run_BLC_ten_by_ten_rand_d1_p1_GPU(self):
		B = BLC.BLC_GPU();  # multiprocessing and GPU version
		B.messup()
		R = BLC.init_R(1, 10, 10, 1, 0, verbose=0)
		ratings = {}; ratings['R'] = R
		B.d = 1; B.p1=1
		U,V,err2,P = B.run_BLC(ratings, verbose=0)
		self.assertEqual(P.shape,(1,10))
		self.assertEqual(U.shape,(1,P.shape[0]))
		self.assertEqual(V.shape,(1,10))
		err = ((P.transpose().dot(U.transpose().dot(V)) - R).A**2).sum()/R.nnz
		self.assertTrue(err<0.01)
		self.assertTrue(abs(err-err2)<0.01)

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(Test_standalone)
	unittest.TextTestRunner(verbosity=2).run(suite)
	suite = unittest.TestLoader().loadTestsFromTestCase(Test_BLCpieces)
	unittest.TextTestRunner(verbosity=2).run(suite)
	suite = unittest.TestLoader().loadTestsFromTestCase(TestBLC)
	unittest.TextTestRunner(verbosity=2).run(suite)
