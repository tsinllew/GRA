#! /home/liu/.pyenv/shims/python

import time
import timeit
import numpy as np
import networkx as nx
import sklearn
import argument
from sklearn import preprocessing
import netdata
import utils
from os.path import join as pjoin
import pickle
import csv
import pickle
import re
import sys
import numpy as np
import networkx as nx
import scipy.io as spio
import emb_mat_decom
from sim_matrix import get_sim_matrix
from os.path import join as pjoin
from smf import SMF

from scipy import linalg as la
from scipy.sparse import linalg as sla
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from scipy.sparse.linalg import svds

import os
import numpy as np
import pickle



sim_type = 'ex_ra'
method = 'smf'
decom_method = 'svd'

args = argument.parse_args()
net_file = args.net_file
print('netfile: {}'.format(net_file))

emb_file = args.emb_file
print('emb_file: {}'.format(emb_file))

net_name = args.net_name
print('net_name: {}'.format(net_name))

emb_dim = args.emb_dim
print('emb_dim: {}'.format(emb_dim))

alpha = args.alpha
print('alpha: {}'.format(alpha))

beta = args.beta
print('beta: {}'.format(beta))

load_S = args.load_S
S_file = args.S_file


network_data = netdata.NetworkData(net_file=net_file, net_name=net_name)
num_nodes = network_data.num_nodes
print('num. nodes: {}, num. edges: {}'.format(network_data.num_nodes, network_data.num_edges))
print('finished network data')


start_time = timeit.default_timer()
model = SMF(emb_dim=emb_dim, alpha=alpha, sim_type=sim_type)
emb = model.learn_embedding(network_data=network_data, beta=beta, sval_sqrt=False, load_S=load_S, S_file=S_file, emb_file=emb_file)

with open(emb_file, 'rb') as f:
	emb = pickle.load(f)

print('total running time:', timeit.default_timer() - start_time)

