#! /home/tsinllew/.pyenv/shims/python

from __future__ import print_function
import time
import timeit
import numpy as np
import networkx as nx
import sklearn
from sklearn import preprocessing
import netdata
import utils
from os.path import join as pjoin
import pickle
import csv
import utils
import re
import sys
import numpy as np
import networkx as nx
import scipy.io as spio
import scipy
from os.path import join as pjoin

from scipy import linalg as la
from scipy.sparse import linalg as sla
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.linalg import logm


def check_validility(S):
	if np.isnan(S).any(): #nan
		raise ValueError('S error: cantains nan')
		sys.exit()
	if np.isinf(S).any(): #inf
		raise ValueError('S error: cantains inf')
		sys.exit()
	if np.min(S) < 0:
		if np.min(S) < -1e-6:
			raise ValueError( 'Similarity Matrix has minus entry. min entry:', np.min(S))
			print( np.unravel_index(S.argmin(), S.shape) )
			print( S.diagonal() )
			sys.exit()
		else:
			S[S<0] = 0



def get_sim_matrix(network_data, sim_type='ex_ra', alpha=0.8, add_selfloop=False):
	network_g = network_data.networkG
	num_nodes = network_data.num_nodes

	if sim_type in ['ex_ra']:
		nsize = num_nodes
		num_edges = network_data.num_edges
		A_sp = nx.adjacency_matrix(network_g, nodelist=sorted(network_g.nodes()))
		A_sp = A_sp.astype(np.float)


		I_sp = scipy.sparse.identity(nsize,dtype=np.float).tocsr()
		if add_selfloop:
			A_sp += I_sp

		diags = A_sp.sum(axis=1)
		D_sp = scipy.sparse.spdiags(diags.flatten(), [0], nsize, nsize, format='csr')
		D_inv_sp = scipy.sparse.spdiags(1/diags.flatten(), [0], nsize, nsize, format='csr')

		SDinv = np.matrix(np.zeros((nsize,nsize), dtype=np.float))
		aADinv = alpha * A_sp * D_inv_sp
		for i in range(1000):
			SDinv_ = aADinv * (SDinv + I_sp)

			if i % 100 == 0:
				err = la.norm(SDinv - SDinv_)
				# print(i, err)

			SDinv = SDinv_
			if err < 1e-10:
				break

		if err >= 1e-10:
			print('S construction error: doesn not converge. Convergence Error: ', err)

		S = SDinv * D_sp

	check_validility(S)

	return np.mat(S)

