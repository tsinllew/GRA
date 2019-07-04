import argparse

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--net_name', nargs='?', default='brazil_flights',
						help='Input network name')

	parser.add_argument('--net_file', nargs='?', default='../network/brazil_flights/brazil_flights.net',
						help='Input network file')

	parser.add_argument('--emb_file', nargs='?', default='../emb/brazil_flights.emb',
						help='Output embedding file')

	parser.add_argument('--S_file', nargs='?', default=None,
						help='Optional iutput similarity file')

	parser.add_argument('--load_S', dest='load_S', action='store_true',
						help='Whether to load a similarity matrix. Default is False.')
	parser.set_defaults(is_bipartite=False)

	parser.add_argument('--emb_dim', type=int, default=120,
						help='Dimension of embeddings. Should be less than the number of nodes. Default is 120.')

	parser.add_argument('--alpha', type=float, default=0.95,
						help='The decay rate parameter alpha.')

	parser.add_argument('--beta', type=float, default=0.0,
						help='The shift number parameter beta.')

	return parser.parse_args()

