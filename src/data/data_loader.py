import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imports import *
from feature_functions import *


@dataclasses.dataclass(frozen=True)
class GraphData:
  features: Sequence[np.ndarray]
  labels: Sequence[np.ndarray]
  adjacencies: Sequence[sp.csr_matrix]
  graph_sizes: Sequence[np.ndarray]


@functools.lru_cache()
def read_labels(DATA_DIR):
  with open(os.path.join(DATA_DIR, "labels.json")) as f:
    return json.load(f)


def read_graph_sizes(DATA_DIR):
  with open(os.path.join(DATA_DIR, "graph_sizes.json")) as f:
    return json.load(f)


def iter_graph(DATA_DIR):
  #     NUM_GRAPHS = len([f for f in os.listdir(DATA_DIR) if f.startswith("graph_")])
  NUM_GRAPHS = len([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])  # After adding graph_size
  for i in range(NUM_GRAPHS):
    filename = os.path.join(DATA_DIR, f"graph_{i:05d}.npz")
    yield nx.from_scipy_sparse_matrix(
      sp.load_npz(filename), create_using=nx.DiGraph)


def convert_networkx_to_adjacency_input(graph):
  adjacency_matrix = nx.to_scipy_sparse_array(graph, format='coo')
  # adjacency_matrix = nx.to_scipy_sparse_matrix(graph, format='coo')
  adjacency_matrix += sp.eye(adjacency_matrix.shape[0])
  return adjacency_matrix


def generate_graph_data(DATA_DIR, feature_list):
  """This generates a dataset for training a GraphNet model.

  Args:
    DATA_DIR: The directory path where raw data saved in.

  Returns:
    An GraphData instance with features, adjacencies and labels.
  """
  ys = np.array(read_labels(DATA_DIR))
  gss = np.array(read_graph_sizes(DATA_DIR))
  # ys = np.array([list(pad(kl, max_degree, 0)) for kl in kls])
  # ys = ys[:, degree_label:degree_label+1]

  features = []
  adjacencies = []

  for graph in iter_graph(DATA_DIR):
    feat_dict = dict()
    for feature in feature_list.keys():
      print(feature)
      feat_dict[feature] = feature_list[feature](graph)

    curr_feature = np.zeros((len(graph), len(feat_dict)))

    for n, node in enumerate(graph.nodes):
      for i, (name, value) in enumerate(feat_dict.items()):
        curr_feature[n, i] = value[node]

    features.append(curr_feature)
    adjacencies.append(convert_networkx_to_adjacency_input(graph))

  return GraphData(features=features, labels=ys, adjacencies=adjacencies, graph_sizes=gss)


@dataclasses.dataclass(frozen=True)
class InputData:
  features: Sequence[np.ndarray]
  labels: Sequence[np.ndarray]
  rows: Sequence[sp.csr_matrix]
  columns: Sequence[sp.csr_matrix]
  edge_types: Sequence[EDGE_TYPE]
  graph_sizes: Sequence[np.ndarray]


def load_input_data(DATA_DIR, feature_list={'constant': constant_feature}, train_fraction=0.8):
  """Loads input data for the specified prediction problem.

  This loads a dataset that can be used with a GraphNet model.

  Returns:
    Three InputData instances with features, rows, cols and labels. They are the
    full/train/test set respectively.
  """

  print(f"Generating data from the directory {DATA_DIR}")
  graph_data = generate_graph_data(DATA_DIR, feature_list)
  features = graph_data.features
  adjacencies = graph_data.adjacencies
  ys = graph_data.labels
  gss = graph_data.graph_sizes

  rows = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies]
  cols = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies]
  edge_types = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies]

  num_training = int(len(ys) * train_fraction)

  features_train = features[:num_training]
  rows_train = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies[:num_training]]
  cols_train = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies[:num_training]]
  edge_types_train = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies[:num_training]]
  ys_train = ys[:num_training]
  gss_train = gss[:num_training]

  features_test = features[num_training:]
  rows_test = [np.array(sp.coo_matrix(a).row, dtype=np.int16) for a in adjacencies[num_training:]]
  cols_test = [np.array(sp.coo_matrix(a).col, dtype=np.int16) for a in adjacencies[num_training:]]
  edge_types_test = [np.array(sp.coo_matrix(a).data, dtype=np.int16) for a in adjacencies[num_training:]]
  ys_test = ys[num_training:]
  gss_test = gss[num_training:]
  return (
    InputData(features=features, labels=ys, rows=rows, columns=cols, edge_types=edge_types, graph_sizes=gss),
    InputData(features=features_train, labels=ys_train, rows=rows_train, columns=cols_train,
              edge_types=edge_types_train, graph_sizes=gss_train),
    InputData(features=features_test, labels=ys_test, rows=rows_test, columns=cols_test,
              edge_types=edge_types_test, graph_sizes=gss_test))


# @markdown As the graphs generally do not have the same number of nodes, and because
# @markdown JAX relies on data shapes being fixed and known upfront, we batch
# @markdown together a set of graphs into a large batch graph that contains each
# @markdown graph as a disconnected component.
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
    b_ys[i, 0] = ys[i]
    root_node = root_nodes[i]
    b_masks[i, root_node, 0] = 1.0

  b_features = b_features.reshape((-1, b_features.shape[-1]))
  b_rows = np.concatenate(b_rows)
  b_cols = np.concatenate(b_cols)

  return b_features, b_rows, b_cols, b_ys, b_masks


def batch_e(features, rows, cols, ys, edge_types):
  """Converts a list of training examples into a batched single graph."""
  batch_size = len(features)
  max_features = max(f.shape[0] for f in features)
  b_features = np.zeros((batch_size, max_features, features[0].shape[1]))
  b_rows = []
  b_cols = []
  b_edge_types = []
  b_ys = np.zeros((batch_size, 1))
  for i in range(batch_size):
    b_features[i, :features[i].shape[0], :] = features[i]
    b_rows.append(rows[i] + i * max_features)
    b_cols.append(cols[i] + i * max_features)
    b_edge_types.append(edge_types[i])
    b_ys[i, 0] = ys[i]

  b_features = b_features.reshape((-1, b_features.shape[-1]))
  b_rows = np.concatenate(b_rows)
  b_cols = np.concatenate(b_cols)
  b_edge_types = np.concatenate(b_edge_types)
  return b_features, b_rows, b_cols, b_ys, b_edge_types
