import json
import itertools
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os

import collections
import dataclasses
import datetime
import enum
import functools
import pickle
import random
import tempfile
from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import optax
import psutil
import seaborn as sns

class EDGE_TYPE():
  SELF_LOOP = 1
  SINGLE_ARROW = 2
  DOUBLE_ARROW = 3
  DASHED_ARROW = 4
