#! /home/tsinllew/.pyenv/shims/python

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
import pickle
import re
import sys
import numpy as np
import networkx as nx
import scipy.io as spio
# import pram_opts
import emb_mat_decom
from sim_matrix import get_sim_matrix
from os.path import join as pjoin
from sklearn.model_selection import train_test_split


from scipy import linalg as la
from scipy.sparse import linalg as sla
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.linalg import logm
from sklearn.decomposition import TruncatedSVD


import os
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import array_str
import utils
import pickle
from sklearn.preprocessing import normalize
import h5py


class SMF(object):

	def __init__(self, emb_dim=120, alpha=0.95, sim_type='ex_ra'):
		self._emb_dim = emb_dim
		self._sim_type = sim_type
		self._alpha = alpha
		self._decom_method = 'svd'


	def get_method_name(self):
		return self._sim_type + '_' + self._decom_method


	def learn_embedding(self, network_data, beta, sval_sqrt, load_S, S_file, emb_file):
		emb_dim = self._emb_dim
		decom_method = self._decom_method
		alpha = self._alpha
		sim_type = self._sim_type

		if sim_type in ['ex_ra']:
			if load_S:
				if not os.path.isfile(S_file):
					raise ValueError('S_file not exist')
				f_h5f = h5py.File(S_file,'r')
				S = f_h5f['sim_mat'][:]
				S = np.mat(S)
				f_h5f.close()
			else:
				S = get_sim_matrix(network_data, sim_type, alpha)

			emb, error = emb_mat_decom.emb_learn(S-beta, decom_method, emb_dim, sval_sqrt)
			emb = preprocessing.normalize(emb, norm='l2', axis=1)

		with open(emb_file, 'wb') as f:
			pickle.dump(emb, f)

		return emb

