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
import scipy
import emb_mat_decom
from os.path import join as pjoin

from scipy import linalg as la
from scipy.sparse import linalg as sla
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from scipy import sparse


import os
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import array_str
import utils
import pickle
from os.path import join as pjoin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



def emb_learn(S, method, emb_dim, sval_sqrt=True, error=True):
	nsize = len(S)

	if method == 'svd':
		svecs_l, svals, svecs_r = randomized_svd(S, n_components=emb_dim, n_iter=5, random_state=1)
		if error:
			part_sum = svals.sum()
			all_sum = svdvals(S).sum()

		if sval_sqrt:
			emb = np.mat(svecs_l) * scipy.sparse.spdiags(np.sqrt(svals), [0], emb_dim, emb_dim, format='csr')
		else:
			emb = np.mat(svecs_l) * scipy.sparse.spdiags(svals, [0], emb_dim, emb_dim, format='csr')

		if error:
			error = 1 - part_sum/all_sum
		else:
			error = None

	return np.array(emb), error
