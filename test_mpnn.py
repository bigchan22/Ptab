#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import enum
import haiku as hk
import jax
import jax.numpy as jnp
import os

# In[2]:


from BH.data_loader import *
from BH.generate_data import *
from training_info import *
from Model_e import Model_e,Direction,Reduction
# from Load_Data import batch
from Train import train,print_accuracies


# In[3]:


#@title Load data

#@markdown Training this model is pretty slow - an hour or so on the free tier colab, but subject to inactivity timeouts and pre-emptions.

#@markdown In order to make it possible to recreate the results from the paper reliably and quickly, we provide several helpers to either speed things up, or reduce the memory footprint:
#@markdown * Pretrained weights - greatly speeds things up by loading the trained model parameters rather than learning from the data
#@markdown * If you are running on a high memory machine (ie *not* on the free colab instance!) the input graph data can be loaded from a pickle (which is faster to load) and kept in memory (faster to re-use, but uses ~12Gb of memory). This makes no difference to training speed (it's only relevant for `generate_graph_data()` and `get_saliency_vectors()`).


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_pretrained_weights = True  #@param{type:"boolean"}
hold_graphs_in_memory = False  #@param{type:"boolean"}

gb = 1024**3
total_memory = psutil.virtual_memory().total / gb
# Less than 20Gb of RAM means we need to do some things slower, but with lower memory impact - in
# particular, we want to allow things to run on the free colab tier.
if total_memory < 20 and hold_graphs_in_memory:
    raise RuntimeError(f"It is unlikely your machine (with {total_memory}Gb) will have enough memory to complete the colab's execution!")

print("Loading input data...")

full_dataset, train_dataset, test_dataset = load_input_data(DIR_PATH)


# In[4]:


# step_size = 0.001
# batch_size = 16

num_classes = 2


model = Model_e(
    num_layers=num_layers,
    num_features=num_features,
    num_classes=num_classes,
    direction=Direction.FORWARD,
    reduction=Reduction.SUM,
    apply_relu_activation=True,
    use_mask=False,
    share=False,
    message_relu=True,
    with_bias=True)

loss_val_gr = jax.value_and_grad(model.loss)
opt_init, opt_update = optax.adam(step_size)


# In[5]:


#num_epochs = 10
trained_params = model.net.init(
    jax.random.PRNGKey(42),
    features=train_dataset.features[0],
    rows=train_dataset.rows[0],
    cols=train_dataset.columns[0],
    batch_size=1,
    edge_types=train_dataset.edge_types[0])
trained_opt_state = opt_init(trained_params)


# In[ ]:

best_acc = None
for ep in range(1, num_epochs + 1):
    tr_data = list(
        zip(
            train_dataset.features,
            train_dataset.rows,
            train_dataset.columns,
            train_dataset.labels,
            train_dataset.edge_types,
        ))
    random.shuffle(tr_data)
    features_train, rows_train, cols_train, ys_train, edge_types_train = zip(
        *tr_data)

    features_train = list(features_train)
    rows_train = list(rows_train)
    cols_train = list(cols_train)
    ys_train = np.array(ys_train)
    edge_types_train = list(edge_types_train)

    for i in range(0, len(features_train), batch_size):
        b_features, b_rows, b_cols, b_ys, b_edges = batch_e(
            features_train[i:i + batch_size],
            rows_train[i:i + batch_size],
            cols_train[i:i + batch_size],
            ys_train[i:i + batch_size],
            edge_types_train[i:i + batch_size],
        )

        trained_params, trained_opt_state, curr_loss = train(
            loss_val_gr,
            opt_update,
            trained_params,
            trained_opt_state,
            b_features,
            b_rows,
            b_cols,
            b_ys,
            b_edges,
        )

        accs = model.accuracy(
            trained_params,
            b_features,
            b_rows,
            b_cols,
            b_ys,
            b_edges,
        )
        print(datetime.datetime.now(),
              f"Iteration {i:4d} | Batch loss {curr_loss:.6f}",
              f"Batch accuracy {accs:.2f}")

    print(datetime.datetime.now(), f"Epoch {ep:2d} completed!")

    # Calculate accuracy across full dataset once per epoch
    print(datetime.datetime.now(), f"Epoch {ep:2d}       | ", end="")
    test_acc = print_accuracies(model,trained_params, test_dataset, train_dataset)
    if best_acc == None or best_acc < test_acc:
        best_acc = test_acc
        if save_trained_weights and best_acc > 0.6:
            with open(PARAM_FILE, 'wb') as f:
                pickle.dump(trained_params, f)

# In[ ]:




