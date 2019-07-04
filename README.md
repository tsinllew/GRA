# GRA

This repository provides a reference implementation of GRA network embedding algorithm as described in the paper:

A general view for network embedding as matrix factorization  
Xin Liu, Tsuyoshi Murata, Kyoung-Sook Kim, Chatchawan Kotarasu and Chenyi Zhuang  
WSDM 2019  

Tested in python2.7 environment  
Pre-requisite:  
numpy,scipy,sklearn,networkx(version=1.11)  

Usage:  
python main.py --net_file '../network/brazil_flights/brazil_flights.net' --emb_file '../emb/brazil_flights.emb' --net_name 'brazil_flights' --emb_dim 120 --alpha 0.95 --beta 0.0

net_file is the input of network data, where each line represent a link, i.e. node0,node1,weight. The node id should start from 0 and increase consecutively  
emb_dim is the dimension of the embeddings  
net_name is the name of the network  
alpha and beta are hyper-parameters  

The output embedding is saved in pickle format.
It can be loaded by  
with open(emb_file, 'rb') as f:  
	emb = pickle.load(f)  
The returned emb is a numpy.ndarray, with size (num_nodes, dim_emb). Each row of emb is the embedding vector for one node.  
