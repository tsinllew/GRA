# GRA

This repository provides a reference implementation of node2vec as described in the paper:

A general view for network embedding as matrix factorization
Xin Liu, Tsuyoshi Murata, Kyoung-Sook Kim, Chatchawan Kotarasu and Chenyi Zhuang
WSDM 2019

Tested in python2.7 environment
Pre-requisite:
numpy,scipy,sklearn,networkx(version=1.11)

Usage:
python main.py --net_file '../network/brazil_flights/brazil_flights.net' --emb_file '../emb/brazil_flights.emb' --net_name 'brazil_flights' --emb_dim 120 --alpha 0.95 --beta 0.0

net_file is the input of network data, which format:
7,77,1.0

29,50,1.0

3,35,1.0

9,84,1.0

... 

Each line represent a link, i.e. node0,node1,weight
the node id should start from 0 and increase consecutively

emb_dim is the dimension of the embeddings
net_name is the name of the network
alpha and beta are hyper-parameters

The output embedding is saved as emb_file
We can load the output by pickle as
with open(emb_file, 'rb') as f:
	emb = pickle.load(f)
The returned emb is a numpy.ndarray, with size (num_nodes, dim_emb). Each row of emb is the embedding vector for one node.
