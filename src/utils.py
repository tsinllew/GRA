# -*- coding: UTF-8 -*-

####################################################################
# Copyright (C) 2017 by LIU XIN. All Rights Reserved.
# Email: tsinllew@gmail.com
####################################################################

import numpy as np
import os
import shutil
from os.path import join as pjoin
import re
import sys
import pickle
import datetime
import functools
import math



def read_config():
	import ConfigParser
	config = ConfigParser.ConfigParser()
	config.read( pjoin(os.path.abspath(os.path.dirname(__file__)), '../config', 'settings.ini') )
	return config


def get_para(section, para, config=read_config()):
	try:
		return config.get(section, para)
	except:
		raise ValueError('get_para function input value error: section, para')


def check_and_clean_dir(dir):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.makedirs(dir)


def workdir_out(path):
	work_dir = pjoin(os.path.abspath(os.path.dirname(__file__)), '..')
	return pjoin(work_dir, path)


def check_dir_to_file(myfile):
	dir_to_file = os.path.dirname(myfile)
	if not os.path.exists(dir_to_file):
		os.makedirs(dir_to_file)


def get_path(type, net_name='karate', method='my', task='label_clas', train_ratio=0.8, tmp_name='tmp', dim=120, add_info=None, data_num=None, sim_type=None, alpha=None):

	suffix_dict = {
	'edge_list': 				'.net',
	'degree_dict': 				'.degree',
	'embedding': 				'.emb',
	'partition':				'.div',
	'bipartite':				'.bipartite',
	'rebuilt':					'.net',
	'label':					'.label',
	'log':						'.log',
	'evl':						'.evl',
	'adj_mat':					'.adjmat',
	'sim_mat':					'.hdf5',
	'linkpred_trainnet_adjmat':	'.adjmat',
	'linkpred_trainnet_openne_edgelist':	'.openne_net',
	'train_net':				'.pickle',
	'train_net':				'.pickle',
	'test_net':					'.pickle',
	'linkpred_data':			'.pickle',
	'labelclas_data':			'.pickle',
	'tmp':						'.tmp'
	}

	if type in ['edge_list', 'degree_dict', 'partition', 'bipartite', 'label', 'adj_mat']:
		path = pjoin('network', net_name)
	elif type in ['linkpred_trainnet_adjmat']:
		path = pjoin('network', net_name, 'linkpred_data', 'train_net')
	elif type in ['linkpred_trainnet_openne_edgelist']:
		path = pjoin('network', net_name, 'linkpred_data', 'train_net')
	elif type in ['linkpred_data']:
		path = pjoin('network', net_name, 'linkpred_data')
	elif type in ['labelclas_data']:
		path = pjoin('network', net_name, 'labelclas_data')
	elif type == 'embedding':
		path = pjoin('emb', task, method, 'dim{0:d}'.format(dim))
	elif type in ['sim_mat']:
		path = pjoin('simmat', net_name)
	elif type == 'rebuilt':
		path = pjoin('rbt', method)
	elif type == 'log':
		path = pjoin('log', method)
	elif type == 'tf_summary':
		path = pjoin('tmp', 'tf_summary', net_name)
	elif type == 'evl':
		path = 'eval'
	elif type == 'tmp':
		path = pjoin('tmp', 'tmp')
	elif type in ['train_net', 'test_net']:
		path = pjoin('network', net_name, 'train_test')

	if type in ['edge_list', 'degree_dict', 'partition', 'bipartite', 'rebuilt', 'label', 'adj_mat', 'log']: # return a file path
		file = net_name + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['embedding']:
		if add_info:
			file = net_name + '_' + add_info + suffix_dict[type]
		else:
			file = net_name + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['linkpred_data', 'linkpred_trainnet_adjmat', 'linkpred_trainnet_openne_edgelist']:
		file = '{0:0.2f}'.format(train_ratio) + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['sim_mat']:
		file = '{0}_alpha{1:0.2f}'.format(sim_type, alpha) + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['labelclas_data']:
		if data_num is None:
			file = '{0:0.2f}'.format(train_ratio) + suffix_dict[type]
		else:
			file = 'time{0}_train{1:0.2f}'.format(data_num,train_ratio) + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['evl']:
		file = task + suffix_dict[type]
		path = pjoin(path, file)
	elif type in ['train_net', 'test_net']:
		if type == 'train_net':
			file = 'train' + '_' + '{0:0.2f}'.format(train_ratio) + suffix_dict[type]
		if type == 'test_net':
			file = 'test' + '_' + '{0:0.2f}'.format(train_ratio) + suffix_dict[type]
		path = pjoin(path, file)
	elif type == 'tmp':
		file = tmp_name + suffix_dict[type]
		path = pjoin(path, file)

	path = workdir_out(path)
	check_dir_to_file(path)
	return path
