#!/usr/bin/env python
# coding: utf-8

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# For training it is strongly encouraged that a GPU is used (eg a local kernel or colab pro). Training on the free colab instance can be done, but the risk of the inactivity timeout or preemption make it less reliable since training for 20 epochs takes around an hour. Ideally 50 epochs would be used to complete training.
#
# Pretrained weights are provided so by default the colab will use these and run from start to finish in a reasonable time and replicate the results from the paper from these saved weights.

# In[ ]:


#@title Install modules
# from IPython.display import clear_output
#
# get_ipython().system('pip install dm-haiku')
# get_ipython().system('pip install jax')
# get_ipython().system('pip install optax')
# clear_output()


# In[ ]:


#@title Imports

import collections
import datetime



import json
import os

import random
import tempfile



import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import optax
import psutil
import seaborn as sns


# In[ ]:


#@title Download data

DATA_DIR = tempfile.mkdtemp()
get_ipython().system('mkdir -p {DATA_DIR}')

print(f"Copying data to {DATA_DIR} - NB this requires ~1.5G of space.")
get_ipython().system('gsutil -m cp "gs://maths_conjectures/representation_theory/*" "{DATA_DIR}/"')

# Extract the graph data.
GRAPH_DIR = os.path.join(DATA_DIR, "graph_data")
get_ipython().system('mkdir -p {GRAPH_DIR}')
get_ipython().system('tar -xzf {DATA_DIR}/graph_data.tar.gz -C {GRAPH_DIR}')

get_ipython().system('echo "Files present:"')
get_ipython().system('ls -lh {DATA_DIR}')
get_ipython().system('du -hs {DATA_DIR}')

with open(os.path.join(DATA_DIR, "graph_index_to_node_index_to_permutation.json"), "rt") as f:
    graph_index_to_node_index_to_permutation = json.load(f)
NUM_GRAPHS = len([f for f in os.listdir(GRAPH_DIR) if f.startswith("graph_")])


# In[ ]:

# In[ ]:



# In[ ]:


#@title Load data

#@markdown Training this model is pretty slow - an hour or so on the free tier colab, but subject to inactivity timeouts and pre-emptions.

#@markdown In order to make it possible to recreate the results from the paper reliably and quickly, we provide several helpers to either speed things up, or reduce the memory footprint:
#@markdown * Pretrained weights - greatly speeds things up by loading the trained model parameters rather than learning from the data
#@markdown * If you are running on a high memory machine (ie *not* on the free colab instance!) the input graph data can be loaded from a pickle (which is faster to load) and kept in memory (faster to re-use, but uses ~12Gb of memory). This makes no difference to training speed (it's only relevant for `generate_graph_data()` and `get_saliency_vectors()`).

use_pretrained_weights = True  #@param{type:"boolean"}
hold_graphs_in_memory = False  #@param{type:"boolean"}

gb = 1024**3
total_memory = psutil.virtual_memory().total / gb
# Less than 20Gb of RAM means we need to do some things slower, but with lower memory impact - in
# particular, we want to allow things to run on the free colab tier.
if total_memory < 20 and hold_graphs_in_memory:
    raise RuntimeError(f"It is unlikely your machine (with {total_memory}Gb) will have enough memory to complete the colab's execution!")

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(degree_label=4)


# The below section defines the model used for predicting a given KL coefficient from an adjacency representation of the Bruhat interval. The model is a version of the Message-Passing Neural Network of Gilmer et al. While there may be other models that can also effectively model this problem, this was chosen in part due to knowledge of the structure of the KL polynomials. We treat the problem of predicting a coefficient as a classification problem, with the number of classes as the largest coefficient observed in the dataset. While this ignores ordering information in the label, we are still able to achieve high accuracies and derive insights from the network.

# In[ ]:


#@title Network Setup

step_size = 0.001
batch_size = 100

num_classes = np.max(train_dataset.labels) + 1
model = Model(
    num_layers=3,
    num_features=64,
    num_classes=num_classes,
    direction=Direction.BOTH,
    reduction=Reduction.SUM,
    apply_relu_activation=True,
    use_mask=False,
    share=False,
    message_relu=True,
    with_bias=True)

loss_val_gr = jax.value_and_grad(model.loss)
opt_init, opt_update = optax.adam(step_size)


def train(params, opt_state, features, rows, cols, ys, masks):
  curr_loss, gradient = loss_val_gr(params, features, rows, cols, ys, masks)
  updates, opt_state = opt_update(gradient, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, opt_state, curr_loss


def compute_accuracies(params_to_evaluate, dataset, batch_size=100):
  total_correct = 0.0
  for i in range(0, len(dataset.features), batch_size):
    b_features, b_rows, b_cols, b_ys, b_masks = batch(
        dataset.features[i:i + batch_size], dataset.rows[i:i + batch_size],
        dataset.columns[i:i + batch_size], dataset.labels[i:i + batch_size],
        dataset.root_nodes[i:i + batch_size])

    accs = model.accuracy(params_to_evaluate, b_features, b_rows, b_cols, b_ys,
                          b_masks)
    total_correct += accs * len(dataset.features[i:i + batch_size])
  return total_correct / len(dataset.features)


def print_accuracies(params_to_evaluate,
                     dataset_test,
                     dataset_train,
                     batch_size=100):
  train_accuracy = compute_accuracies(
      params_to_evaluate, dataset=train_dataset, batch_size=batch_size)
  test_accuracy = compute_accuracies(
      params_to_evaluate, dataset=test_dataset, batch_size=batch_size)

  combined_accuracy = np.average(
      [train_accuracy, test_accuracy],
      weights=[len(dataset_train.features),
               len(dataset_test.features)])
  print(f'Train accuracy: {train_accuracy:.3f} | '
        f'Test accuracy: {test_accuracy:.3f} | '
        f'Combined accuracy: {combined_accuracy:.3f}')


# To replicate the Figure 3a from the paper, it is sufficient to use pre-trained set of parameters which were trained for 100 epochs on the 4th degree coefficient of S9. To do so, leave the above `use_pretrained_weights` set to `True` and the (**much** slower) training loop can be skipped.
#
# To replicate the results from scratch, set `use_pretrained_weights=False` and perform training from a fresh set of parameters. The final results should be visible after a large number of epochs, and although full convergence is usually achieved before 100 epochs it is still expected to take an hour on GPU and even longer on CPU.

# In[ ]:


#@title Perform training / Load pretrained weights

if use_pretrained_weights:
  print("Loading pre-trained weights")
  flat_trained_params = jax.numpy.load(
      os.path.join(DATA_DIR, "trained_params.npz"))
  trained_params = collections.defaultdict(dict)
  for key, array in flat_trained_params.items():
    layer, weight_or_bias = key.split()
    assert weight_or_bias in ("w", "b"), weight_or_bias
    assert "linear" in layer, layer
    trained_params[layer][weight_or_bias] = array
  trained_params = dict(trained_params)
else:
  num_epochs = 20
  trained_params = model.net.init(
      jax.random.PRNGKey(42),
      features=train_dataset.features[0],
      rows=train_dataset.rows[0],
      cols=train_dataset.columns[0],
      batch_size=1,
      masks=train_dataset.features[0][np.newaxis, :, :])
  trained_opt_state = opt_init(trained_params)

  for ep in range(1, num_epochs + 1):
    tr_data = list(
        zip(
            train_dataset.features,
            train_dataset.rows,
            train_dataset.columns,
            train_dataset.labels,
            train_dataset.root_nodes,
        ))
    random.shuffle(tr_data)
    features_train, rows_train, cols_train, ys_train, root_nodes_train = zip(
        *tr_data)

    features_train = list(features_train)
    rows_train = list(rows_train)
    cols_train = list(cols_train)
    ys_train = np.array(ys_train)
    root_nodes_train = list(root_nodes_train)

    for i in range(0, len(features_train), batch_size):
      b_features, b_rows, b_cols, b_ys, b_masks = batch(
          features_train[i:i + batch_size],
          rows_train[i:i + batch_size],
          cols_train[i:i + batch_size],
          ys_train[i:i + batch_size],
          root_nodes_train[i:i + batch_size],
      )

      trained_params, trained_opt_state, curr_loss = train(
          trained_params,
          trained_opt_state,
          b_features,
          b_rows,
          b_cols,
          b_ys,
          b_masks,
      )

      accs = model.accuracy(
          trained_params,
          b_features,
          b_rows,
          b_cols,
          b_ys,
          b_masks,
      )
      print(datetime.datetime.now(),
            f"Iteration {i:4d} | Batch loss {curr_loss:.6f}",
            f"Batch accuracy {accs:.2f}")

    print(datetime.datetime.now(), f"Epoch {ep:2d} completed!")

    # Calculate accuracy across full dataset once per epoch
    print(datetime.datetime.now(), f"Epoch {ep:2d}       | ", end="")
    print_accuracies(trained_params, test_dataset, train_dataset)


# In[ ]:


#@title Print model accuracies
#@markdown Baseline accuracy should be ~88%; trained accuracy should be ~98%.

#@markdown If only 20 epochs are trained for (as is the default setting above
#@markdown for training from scratch), the overall accuracy will be between
#@markdown the two, near 95%.
print('Baseline accuracy', 1 - np.mean(train_dataset.labels))
print_accuracies(trained_params, test_dataset, train_dataset)


# In[ ]:


#@title Calculate salience and aggregate by edge labels
def get_salience_vectors(salience_fn, params, full_dataset):
  salient_features_arr = []
  for i in range(0, len(full_dataset.features), batch_size):
    b_features, b_rows, b_cols, b_ys, b_masks = batch(
        full_dataset.features[i:i + batch_size],
        full_dataset.rows[i:i + batch_size],
        full_dataset.columns[i:i + batch_size],
        full_dataset.labels[i:i + batch_size],
        full_dataset.root_nodes[i:i + batch_size],
    )
    salient_features = salience_fn(params, b_features, b_rows, b_cols, b_ys,
                                   b_masks)
    effective_batch_size = len(full_dataset.features[i:i + batch_size])
    salient_features_arr.extend(
        np.reshape(salient_features, [effective_batch_size, -1, 2]))
  return salient_features_arr


def aggregate_by_edges(salient_features_arr, cutoff, ys):
  refl_count = {
      'salient_all': collections.defaultdict(int),
      'all': collections.defaultdict(int)
  }
  for graph_index, (graph, saliency, label) in enumerate(
      zip(iter_graph(), salient_features_arr, ys)):
    [salient_nodes] = np.where(np.linalg.norm(saliency, axis=1) > cutoff)
    subgraph = graph.subgraph(salient_nodes)
    for reflection in get_reflections(graph_index, graph):
      refl_count['all'][reflection] += 1
    for reflection in get_reflections(graph_index, subgraph):
      refl_count['salient_all'][reflection] += 1

  norm_refl_mat = {}
  for title, counts in refl_count.items():
    reflection_mat = np.zeros((9, 9))
    # Loop over the upper triangle.
    for i in range(9):
      for j in range(i + 1, 9):
        count = counts[(i, j)] + counts[(j, i)]
        reflection_mat[i, j] = count
        reflection_mat[j, i] = count
    norm_refl_mat[title] = reflection_mat / reflection_mat.sum()

  return refl_count, norm_refl_mat


def get_reflections(graph_index, graph):
  node_index_to_permutation = graph_index_to_node_index_to_permutation[str(
      graph_index)]
  for permutation_x, permutation_y in graph.edges():
    if np.isscalar(permutation_x):
      # If the data was loaded as compressed sci-py arrays, the permutations
      # need to be looked up by index in the data loaded separate from JSON.
      permutation_x = node_index_to_permutation[str(permutation_x)]
      permutation_y = node_index_to_permutation[str(permutation_y)]
    yield tuple(i for i, (x, y) in enumerate(zip(permutation_x, permutation_y))
                if x != y)


print('Computing saliences...')
salience_fn = jax.jit(jax.grad(lambda *args: jnp.sum(model.loss(*args)), 1))
salient_features_arr = get_salience_vectors(salience_fn, trained_params,
                                            full_dataset)
saliencies = np.linalg.norm(
    np.concatenate(salient_features_arr, axis=0), axis=1)

print('Aggregating by edges...')
cutoff = np.percentile(saliencies, 99)
refl_count, norm_refl_mat = aggregate_by_edges(salient_features_arr, cutoff,
                                               full_dataset.labels)


# The final cell replicates Figure 3a from the paper - it shows the relative frequency of different edge types in salient subgraphs compared with the frequency across the full dataset.

# In[ ]:


#@title Plot edge attribution

font = {'family': 'normal', 'weight': 'bold', 'size': 18}

matplotlib.rc('font', **font)
sns.set_style('ticks')

np.fill_diagonal(norm_refl_mat['all'], 1)  # Avoid 0/0 warning.
change_grid = ((norm_refl_mat['salient_all'] - norm_refl_mat['all']) /
               norm_refl_mat['all'] * 100)

f, ax = plt.subplots(figsize=(10, 10))
ax = sns.heatmap(
    change_grid,
    mask=np.triu(np.ones_like(change_grid)),
    center=0,
    square=True,
    cmap='RdBu',
    cbar_kws={'shrink': .82},
    ax=ax,
    vmin=-50,
    vmax=50)

ax.set_ylabel('1st reflection index')
ax.set_xlabel('2nd reflection index')
sns.despine()

plt.show()

