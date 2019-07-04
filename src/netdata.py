import sys
import collections
import numpy as np
import networkx as nx
from random import randint
from networkx.algorithms import bipartite
import random
import pickle
import utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class NetworkData():
	def _read_network(self, net_file, delimiter, is_weighted, is_directed, is_bipartite, bipartite_file=None):
		if is_weighted:
			networkG = nx.read_edgelist(path=net_file, delimiter=delimiter, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		else:
			networkG = nx.read_edgelist(path=net_file, delimiter=delimiter, nodetype=int, create_using=nx.DiGraph())
			for edge in networkG.edges():
				networkG[edge[0]][edge[1]]['weight'] = 1
		if not is_directed:
			networkG = networkG.to_undirected()
		return networkG


	def __init__(
		self,
		net_file = None,
		delimiter = ',',
		nx_graph = None,
		net_name = None,
		search_talbe_size = np.int32(1e4),
		sim_dicts_in_memory = False,
		max_ratio_nbrs = 0.3,
		min_ratio_spls = 0.3,
		is_weighted = True,
		is_directed = False,
		is_bipartite = False,
		bipartite_file = None
		):
		if not (nx_graph or net_file) or (nx_graph and net_file):
			raise ValueError('NetworkData init function value error: net_file, nx_graph')
		else:
			if not nx_graph:
				self.networkG = self._read_network(net_file=net_file, delimiter=delimiter, is_weighted=is_weighted, is_directed=is_directed, is_bipartite=is_bipartite, bipartite_file=bipartite_file)
			else:
				self.networkG = nx_graph
		self.is_directed = is_directed
		self.is_bipartite = is_bipartite
		self.net_name = net_name
		self.num_nodes = self.networkG.number_of_nodes()
		self.num_edges = self.networkG.number_of_edges()
