import dataclasses


from typing import Sequence

import itertools
import functools
import os
import pickle

import networkx as nx
import numpy as np
import scipy.sparse as sp

#@title Data loading functions

#@markdown These functions load in previously generated Bruhat intervals (NetworkX graphs) and
#@markdown their associated KL polynomials (a list of integer coefficients), converting them
#@markdown into an adjacency matrix format and single label that is appropriate for our JAX model.
#@markdown The label is the degree-label'th coeefficient from the KL polynomial. See
#@markdown generate_graph_data() for details.

train_fraction = .8


def pad(iterable, size, padding=None):
  return itertools.islice(pad_infinite(iterable, padding), size)


def pad_infinite(iterable, padding=None):
  return itertools.chain(iterable, itertools.repeat(padding))


def convert_networkx_to_adjacency_input(graph):
  adjacency_matrix = nx.to_scipy_sparse_matrix(graph, format='coo')
  adjacency_matrix += sp.eye(adjacency_matrix.shape[0])
  return adjacency_matrix


@dataclasses.dataclass(frozen=True)
class GraphData:
  features: Sequence[np.ndarray]
  labels: Sequence[np.ndarray]
  adjacencies: Sequence[sp.csr_matrix]


def generate_graph_data(degree_label):
  """Generate dataset for training GraphNet model on KL data.

  This generates a dataset for training a GraphNet model.

  Args:
    degree_label: The polynomial coefficient to use as the label.

  Returns:
    An GraphData instance with features, adjacencies and labels.
  """
  kls = read_kl_coefficients()
  max_degree = max(len(kl) for kl in kls)

  ys = np.array([list(pad(kl, max_degree, 0)) for kl in kls])
  ys = ys[:, degree_label:degree_label+1]

  features = []

  for graph in iter_graph():
    feat_dict = {
        'in_centrality': nx.in_degree_centrality(graph),
        'out_centrality': nx.out_degree_centrality(graph),
    }

    curr_feature = np.zeros((len(graph), len(feat_dict)))

    for n, perm in enumerate(graph.nodes):
      for i, (name, value) in enumerate(feat_dict.items()):
        curr_feature[n,i] = value[perm]

    features.append(curr_feature)
  adjacencies = [convert_networkx_to_adjacency_input(g) for g in graphs]

  return GraphData(features=features, labels=ys, adjacencies=adjacencies)


@functools.lru_cache()
def load_graphs_from_pickle():
  assert hold_graphs_in_memory, "Should only load data from the pickle if 'hold_graphs_in_memory' is True"
  with open(os.path.join(DATA_DIR, 'bruhat_data_S9.pickle'), 'rb') as ifile:
    unused_interval_spec, unused_interval_lengths, graphs, unused_kls = pickle.load(ifile)
  return graphs


def iter_graph():
  if hold_graphs_in_memory:
    yield from load_graphs_from_pickle()
  else:
    for i in range(NUM_GRAPHS):
      filename = os.path.join(GRAPH_DIR, f"graph_{i:04d}.npz")
      yield nx.from_scipy_sparse_matrix(
          sp.load_npz(filename), create_using=nx.DiGraph)


@functools.lru_cache()
def read_kl_coefficients():
  with open(os.path.join(GRAPH_DIR, "kl_coefficients.json")) as f:
    return json.load(f)


def get_root_node(col):
  return np.bincount(col).argmin()


@dataclasses.dataclass(frozen=True)
class InputData:
  features: Sequence[np.ndarray]
  labels: Sequence[np.ndarray]
  rows: Sequence[sp.csr_matrix]
  columns: Sequence[sp.csr_matrix]
  root_nodes: Sequence[int]


def load_input_data(degree_label=1):
  """Loads input data for the specified prediction problem.

  This loads a dataset that can be used with a GraphNet model. The Bruhat
  intervals are taken from the dataset of intervals in S9 and the label
  is the coefficient of specified degree.

  The datasets are cached, and only regenerated when not found on disk.

  Args:
    degree_label: the polynomial coefficient to use as the label.
  Returns:
    Three InputData instances with features, rows, cols and labels. They are the
    full/train/test set respectively.
  """
  input_data_cache_dir = os.path.join(DATA_DIR, f"input_data_{degree_label}")

  # Extract from .tar if not already done.
  tar_path = f"{DATA_DIR}/S9_{degree_label}.tar.gz"
  cache_dir = os.path.join(DATA_DIR, f"input_data_{degree_label}")
  if os.path.exists(tar_path) and not os.path.exists(cache_dir):
    print(f"Extracting input files from {tar_path}")
    get_ipython().system('mkdir {cache_dir}')
    get_ipython().system('tar -xzf {tar_path} -C {cache_dir}')

  # Load from cache for either extracted-tar or a previously computed run.
  if os.path.exists(input_data_cache_dir):
    print(f"Loading np arrays from directory: '{input_data_cache_dir}'", flush=True)
    # Load adj
    adjacencies = [sp.load_npz(os.path.join(input_data_cache_dir, filename))
                   for filename in sorted(os.listdir(input_data_cache_dir))
                   if not filename.endswith("arrays.npz")]
    # Load np arrays
    with np.load(os.path.join(input_data_cache_dir, "arrays.npz")) as data:
      ys = data["labels"]
      features = [data[f"feature_{i:04d}"] for i in range(len(adjacencies))]
    print("Data loaded from cache.", flush=True)
  else:
    print(f"Generating data for degree_label {degree_label} and caching (~1m to generate)", flush=True)
    graph_data = generate_graph_data(degree_label)
    features = graph_data.features
    adjacencies = graph_data.adjacencies
    ys = graph_data.labels

    # Save to disk to save time in future:
    get_ipython().system('mkdir {input_data_cache_dir}')
    np.savez(os.path.join(input_data_cache_dir, "arrays.npz"),
             **{f"feature_{i:04d}": f for i, f in enumerate(features)}, labels=ys)
    for i, adj in enumerate(adjacencies):
      sp.save_npz(os.path.join(input_data_cache_dir, f"adj_{i:04d}.npz"), adj)
    print(f"Data cached to directory {input_data_cache_dir}; future runs should be much faster!")

  rows = [sp.coo_matrix(a).row for a in adjacencies]
  cols = [sp.coo_matrix(a).col for a in adjacencies]
  root_nodes = [get_root_node(col) for col in cols]

  num_training = int(len(ys) * train_fraction)

  features_train = features[:num_training]
  rows_train = [sp.coo_matrix(a).row for a in adjacencies[:num_training]]
  cols_train = [sp.coo_matrix(a).col for a in adjacencies[:num_training]]
  ys_train = ys[:num_training]
  root_nodes_train = root_nodes[:num_training]

  features_test = features[num_training:]
  rows_test = [sp.coo_matrix(a).row for a in adjacencies[num_training:]]
  cols_test = [sp.coo_matrix(a).col for a in adjacencies[num_training:]]
  ys_test = ys[num_training:]
  root_nodes_test = root_nodes[num_training:]
  return (
      InputData(features=features, rows=rows, columns=cols, labels=ys, root_nodes=root_nodes),
      InputData(features=features_train, rows=rows_train, columns=cols_train, labels=ys_train, root_nodes=root_nodes_train),
      InputData(features=features_test, rows=rows_test, columns=cols_test, labels=ys_test, root_nodes=root_nodes_test))


#@markdown As the graphs generally do not have the same number of nodes, and because
#@markdown JAX relies on data shapes being fixed and known upfront, we batch
#@markdown together a set of graphs into a large batch graph that contains each
#@markdown graph as a disconnected component.
def batch(features, rows, cols, ys, root_nodes):
  """Converts a list of training examples into a batched single graph."""
  batch_size = len(features)
  max_features = max(f.shape[0] for f in features)
  b_features = np.zeros((batch_size, max_features, features[0].shape[1]))
  b_rows = []
  b_cols = []
  b_ys = np.zeros((batch_size, 1))
  b_masks = np.zeros((batch_size, max_features, 1))
  for i in range(batch_size):
    b_features[i, :features[i].shape[0], :] = features[i]
    b_rows.append(rows[i] + i * max_features)
    b_cols.append(cols[i] + i * max_features)
    b_ys[i, 0] = ys[i, 0]
    root_node = root_nodes[i]
    b_masks[i, root_node, 0] = 1.0

  b_features = b_features.reshape((-1, b_features.shape[-1]))
  b_rows = np.concatenate(b_rows)
  b_cols = np.concatenate(b_cols)

  return b_features, b_rows, b_cols, b_ys, b_masks

