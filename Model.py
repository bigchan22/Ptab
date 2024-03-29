import functools
import enum
import haiku as hk
import jax
import jax.numpy as jnp

#@title MPNN model definition code and helper functions


#
class Direction(enum.Enum):
  FORWARD = enum.auto()
  BACKWARD = enum.auto()
  BOTH = enum.auto()


class Reduction(enum.Enum):
  SUM = enum.auto()
  MAX = enum.auto()


class MPNN(hk.Module):
  """Sparse Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __init__(
      self,
      *,
      out_size: int,
      mid_size: int,
      activation,
      direction: Direction,
      residual: bool,
      reduction: Reduction,
      message_relu: bool,
      with_bias: bool,
  ):
    """Build MPNN layer.

    Args:
      out_size: Output width of the network.
      mid_size: Width of the hidden layer.
      activation: Activation function to use before the layer output.
      direction: Direction of message passing. See Direction Enum.
      residual: Whether to use resiudal connections.
      reduction: Reduction function to aggregate messages at nodes. See
        Reduction enum.
      message_relu: Whether to apply a relu on each message.
      with_bias: Whether to add biases in linear layers.

    Returns:
      The output of the MPNN layer.
    """
    super().__init__(name='mpnn_aggr')
    self.mid_size = out_size if mid_size is None else mid_size
    self.out_size = out_size
    self.activation = activation
    self.direction = direction
    self.reduction = reduction
    self.residual = residual
    self.message_relu = message_relu
    self.with_bias = with_bias

    @jax.jit
    def jax_coo_sum(rows, cols, msg_in, msg_out):
      msg_vect = msg_in[rows] + msg_out[cols]
      if message_relu:
        msg_vect = jax.nn.relu(msg_vect)
      return jnp.zeros_like(msg_out).at[rows].add(msg_vect)

    @jax.jit
    def jax_coo_max(rows, cols, msg_in, msg_out):
      msg_vect = msg_in[rows] + msg_out[cols]
      if message_relu:
        msg_vect = jax.nn.relu(msg_vect)
      return jnp.zeros_like(msg_in).at[rows].max(msg_vect)

    self.jax_coo_sum = jax_coo_sum
    self.jax_coo_max = jax_coo_max

  def __call__(self, features, rows, cols):
    if self.direction == Direction.FORWARD or self.direction == Direction.BOTH:
      m1_1 = hk.Linear(self.mid_size, with_bias=self.with_bias)
      m2_1 = hk.Linear(self.mid_size, with_bias=self.with_bias)
      msg_1_1 = m1_1(features)
      msg_2_1 = m2_1(features)
    if self.direction == Direction.BACKWARD or self.direction == Direction.BOTH:
      m1_2 = hk.Linear(self.mid_size, with_bias=self.with_bias)
      m2_2 = hk.Linear(self.mid_size, with_bias=self.with_bias)
      msg_1_2 = m1_2(features)
      msg_2_2 = m2_2(features)

    o2 = hk.Linear(self.out_size, with_bias=self.with_bias)

    if self.reduction == Reduction.MAX:
      reduction = self.jax_coo_max
    elif self.reduction == Reduction.SUM:
      reduction = self.jax_coo_sum
    else:
      raise ValueError('Unknown reduction %s' % self.reduction)

    if self.direction == Direction.FORWARD:
      msgs = reduction(rows, cols, msg_1_1, msg_2_1)
    elif self.direction == Direction.BACKWARD:
      msgs = reduction(cols, rows, msg_1_2, msg_2_2)
    elif self.direction == Direction.BOTH:
      msgs_1 = reduction(rows, cols, msg_1_1, msg_2_1)
      msgs_2 = reduction(cols, rows, msg_1_2, msg_2_2)
      msgs = jnp.concatenate([msgs_1, msgs_2], axis=-1)
      pass
    else:
      raise ValueError('Unknown direction %s' % self.direction)

    h_2 = o2(msgs)
    if self.residual:
      o1 = hk.Linear(self.out_size, with_bias=self.with_bias)
      h_1 = o1(features)
      network_output = h_1 + h_2
    else:
      network_output = h_2

    if self.activation is not None:
      network_output = self.activation(network_output)

    return network_output


class Model:

  def __init__(
      self,
      *,
      num_layers: int,
      num_features: int,
      num_classes: int,
      direction: Direction,
      reduction: Reduction,
      apply_relu_activation: bool,
      use_mask: bool,
      share: bool,
      message_relu: bool,
      with_bias: bool,
  ):
    """Get the jax model function and associated functions.

    Args:
      num_layers: The number of layers in the GraphNet - equivalently the number
        of propagation steps.
      num_features: The dimension of the hidden layers / messages.
      num_classes: The number of target classes.
      direction: Edges to pass messages along, see Direction enum.
      reduction: The reduction operation to be used to aggregate messages at
        each node at each step. See Reduction enum.
      apply_relu_activation: Whether to apply a relu at the end of each
        propogration step.
      use_mask: Boolean; should a masked prediction in central node be
        performed?
      share: Boolean; should the GNN layers be shared?
      message_relu: Boolean; should a ReLU be used in the message function?
      with_bias: Boolean; should the linear layers have bias?
    """
    self._num_layers = num_layers
    self._num_features = num_features
    self._num_classes = num_classes
    self._direction = direction
    self._reduction = reduction
    self._apply_relu_activation = apply_relu_activation
    self._use_mask = use_mask
    self._share = share
    self._message_relu = message_relu
    self._with_bias = with_bias

  def _kl_net(self, features, rows, cols, batch_size, masks):
    in_enc = hk.Linear(self._num_features)

    if self._apply_relu_activation:
      activation_fn = jax.nn.relu
    else:
      activation_fn = lambda net: net

    gnns = []
    for i in range(self._num_layers):
      if i == 0 or not self._share:
        gnns.append(
            MPNN(
                out_size=self._num_features,
                mid_size=None,
                direction=self._direction,
                reduction=self._reduction,
                activation=activation_fn,
                message_relu=self._message_relu,
                with_bias=self._with_bias,
                residual=True))
      else:
        gnns.append(gnns[-1])

    out_enc = hk.Linear(self._num_classes, with_bias=self._with_bias)

    hiddens = []
    hidden = in_enc(features)
    hiddens.append(jnp.reshape(hidden, (batch_size, -1, self._num_features)))
    for gnn in gnns:
      hidden = gnn(hidden, rows, cols)
      hiddens.append(jnp.reshape(hidden, (batch_size, -1, self._num_features)))

    hidden = jnp.reshape(hidden, (batch_size, -1, self._num_features))

    if self._use_mask:
      h_bar = jnp.sum(hidden * masks, axis=1)
    else:
      h_bar = jnp.max(hidden, axis=1)

    lgts = out_enc(h_bar)

    return hiddens, lgts

  @property
  def net(self):
    return hk.transform(self._kl_net)

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params, features, rows, cols, ys, masks):
    _, lgts = self.net.apply(params, None, features, rows, cols, ys.shape[0],
                             masks)
    return -jnp.mean(
        jax.nn.log_softmax(lgts) *
        jnp.squeeze(jax.nn.one_hot(ys, self._num_classes), 1))

  @functools.partial(jax.jit, static_argnums=(0,))
  def accuracy(self, params, features, rows, cols, ys, masks):
    _, lgts = self.net.apply(params, None, features, rows, cols, ys.shape[0],
                             masks)
    pred = jnp.argmax(lgts, axis=-1)
    true_vals = jnp.squeeze(ys, axis=1)
    acc = jnp.mean(pred == true_vals)
    return acc
