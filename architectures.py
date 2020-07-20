# Copyright 2020 Jacob I. Monroe, NIST Employee  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of neural network architectures used for constructing VAEs for images.
Adapted from disentanglement_lib https://github.com/google-research/disentanglement_lib"""
import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class SampleLatent(tf.keras.layers.Layer):
  """Samples from the Gaussian distribution defined by z_mean and z_logvar."""

  def __init__(self, name='sampler', **kwargs):
    super(SampleLatent, self).__init__(name=name, **kwargs)

  def call(self, z_mean, z_logvar):
    return tf.add(z_mean,
                  tf.exp(z_logvar / 2.0) * tf.random.normal(tf.shape(z_mean), 0, 1)
                 )


class FCEncoder(tf.keras.layers.Layer):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl).

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """

  def __init__(self, num_latent, name='encoder', hidden_dim=1200,
               kernel_initializer='glorot_uniform', **kwargs):
    super(FCEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.flattened = tf.keras.layers.Flatten()
    self.e1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e1",
                                    kernel_initializer=self.kernel_initializer)
    self.e2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e2",
                                    kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(num_latent, activation=None,
                                       kernel_initializer=self.kernel_initializer)
    self.log_var = tf.keras.layers.Dense(num_latent, activation=None,
                                         kernel_initializer=self.kernel_initializer)

  def call(self, input_tensor):
    flattened_out = self.flattened(input_tensor)
    e1_out = self.e1(flattened_out)
    e2_out = self.e2(e1_out)
    means_out = self.means(e2_out)
    log_var_out = self.log_var(e2_out)
    return means_out, log_var_out


class ConvEncoder(tf.keras.layers.Layer):
  """Convolutional encoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    input_tensor: Input tensor of shape (batch_size, 64, 64, num_channels) to
      build encoder on.
    num_latent: Number of latent variables to output.

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """

  def __init__(self, num_latent, name='encoder',
               kernel_initializer='glorot_uniform', **kwargs):
    super(ConvEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.kernel_initializer = kernel_initializer
    self.e1 = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=4,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e1",
                                     kernel_initializer=self.kernel_initializer,
                                    )
    self.e2 = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=4,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e2",
                                     kernel_initializer=self.kernel_initializer,
                                    )
    self.e3 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=2,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e3",
                                     kernel_initializer=self.kernel_initializer,
                                    )
    self.e4 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=2,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e4",
                                     kernel_initializer=self.kernel_initializer,
                                    )
    self.flat_e4 = tf.keras.layers.Flatten()
    self.e5 = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="e5",
                                    kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(num_latent, activation=None, name="means",
                                       kernel_initializer=self.kernel_initializer)
    self.log_var = tf.keras.layers.Dense(num_latent, activation=None, name="log_var",
                                         kernel_initializer=self.kernel_initializer)

  def call(self, input_tensor):
    e1_out = self.e1(input_tensor)
    e2_out = self.e2(e1_out)
    e3_out = self.e3(e2_out)
    e4_out = self.e4(e3_out)
    flat_e4_out = self.flat_e4(e4_out)
    e5_out = self.e5(flat_e4_out)
    means_out = self.means(e5_out)
    log_var_out = self.log_var(e5_out)
    return means_out, log_var_out


class FCDecoder(tf.keras.layers.Layer):
  """Fully connected encoder used in beta-VAE paper for the dSprites data.

  Based on row 1 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor to connect decoder to.
    out_shape: Shape of the data.

  Returns:
    Output tensor of shape (None, 64, 64, num_channels) with the [0,1] pixel
    intensities.
  """

  def __init__(self, out_shape, name='decoder',
               hidden_dim=1200,
               kernel_initializer='glorot_uniform',
               return_vars=False, **kwargs):
    super(FCDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.hidden_dim = hidden_dim
    self.kernel_initializer=kernel_initializer
    self.return_vars = return_vars
    self.d1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    self.d2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    #self.d3 = tf.keras.layers.Dense(1200, activation=tf.nn.tanh,
    #                                kernel_initializer=self.kernel_initializer)
    #self.d4 = tf.keras.layers.Dense(np.prod(out_shape),
    #                                kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(np.prod(out_shape), activation=None,
                                       kernel_initializer=self.kernel_initializer)
    if self.return_vars:
      #Set up to return variances in addition to means (don't assume all std are 1)
      #For lattice gas, really returning logits
      #(but once convert to Bernoulli probability can think of as mean)
      #Anyway, for a lattice gas, returning variances does not make sense
      self.log_var = tf.keras.layers.Dense(np.prod(out_shape), activation=None,
                                           kernel_initializer=self.kernel_initializer)

  def call(self, latent_tensor):
    d1_out = self.d1(latent_tensor)
    d2_out = self.d2(d1_out)
    #d3_out = self.d3(d2_out)
    #d4_out = self.d4(d3_out)
    #return tf.reshape(d4_out, shape=(-1,) + self.out_shape)
    means_out = tf.reshape(self.means(d2_out), shape=(-1,)+self.out_shape)
    if self.return_vars:
      log_var_out = tf.reshape(self.log_var(d2_out), shape=(-1,)+self.out_shape)
      return means_out, log_var_out
    else:
      return means_out


class DeconvDecoder(tf.keras.layers.Layer):
  """Convolutional decoder used in beta-VAE paper for the chairs data.

  Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework"
  (https://openreview.net/forum?id=Sy2fzU9gl)

  Args:
    latent_tensor: Input tensor of shape (batch_size,) to connect decoder to.
    out_shape: Shape of the data.

  Returns:
    Output tensor of shape (batch_size, 64, 64, num_channels) with the [0,1]
      pixel intensities.
  """

  def __init__(self, out_shape, name='decoder', 
               kernel_initializer='glorot_uniform', **kwargs):
    super(DeconvDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.kernel_initializer = kernel_initializer
    self.d1 = tf.keras.layers.Dense(256, activation=tf.nn.relu,
                                    kernel_initializer=self.kernel_initializer)
    self.d2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu,
                                    kernel_initializer=self.kernel_initializer)
    self.d3 = tf.keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                              kernel_initializer=self.kernel_initializer,
                                             )
    self.d4 = tf.keras.layers.Conv2DTranspose(filters=32,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                              kernel_initializer=self.kernel_initializer,
                                             )
    self.d5 = tf.keras.layers.Conv2DTranspose(filters=32,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                              kernel_initializer=self.kernel_initializer,
                                             )
    self.d6 = tf.keras.layers.Conv2DTranspose(filters=out_shape[2],
                                              kernel_size=4,
                                              strides=2,
                                              padding="same",
                                              kernel_initializer=self.kernel_initializer,
                                             )

  def call(self, latent_tensor):
    d1_out = self.d1(latent_tensor)
    d2_out = self.d2(d1_out)
    d2_reshaped = tf.reshape(d2_out, shape=[-1, 4, 4, 64])
    d3_out = self.d3(d2_reshaped)
    d4_out = self.d4(d3_out)
    d5_out = self.d5(d4_out)
    d6_out = self.d6(d5_out)
    return tf.reshape(d6_out, (-1,) + self.out_shape)


class FCEncoderFlow(tf.keras.layers.Layer):
  """Fully connected encoder modified to output flow parameters
  """

  def __init__(self, num_latent, name='encoder',
               hidden_dim = 1200,
               kernel_initializer='glorot_uniform',
               flow_net_params={'num_hidden':2, 'hidden_dim':200, 'nvp_split':False},
               flow_mat_rank=1, **kwargs):
    super(FCEncoderFlow, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.flow_num_hidden = flow_net_params['num_hidden']
    self.flow_hidden_dim = flow_net_params['hidden_dim']
    self.flow_nvp_split = flow_net_params['nvp_split']
    self.flow_mat_rank = flow_mat_rank
    self.flattened = tf.keras.layers.Flatten()
    self.e1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e1",
                                    kernel_initializer=self.kernel_initializer)
    self.e2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e2",
                                    kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(num_latent, activation=None, name='means',
                                       kernel_initializer=self.kernel_initializer)
    self.log_var = tf.keras.layers.Dense(num_latent, activation=None, name='logvar',
                                         kernel_initializer=self.kernel_initializer)
    self.flow_U = []
    self.flow_V = []
    self.flow_b = []
    self.flow_input_dims = [self.num_latent,] + self.flow_num_hidden*[self.flow_hidden_dim]
    if self.flow_nvp_split:
      self.flow_output_dims = self.flow_num_hidden*[self.flow_hidden_dim] + [self.num_latent*2,]
    else:
      self.flow_output_dims = self.flow_num_hidden*[self.flow_hidden_dim] + [self.num_latent,]
    for l in range(self.flow_num_hidden+1):
      self.flow_U.append(tf.keras.layers.Dense(self.flow_output_dims[l]*self.flow_mat_rank,
                                               activation=None,))
                                               #kernel_intializer=self.kernel_initializer))
      self.flow_V.append(tf.keras.layers.Dense(self.flow_input_dims[l]*self.flow_mat_rank,
                                               activation=None,))
                                               #kernel_intializer=self.kernel_initializer))
      self.flow_b.append(tf.keras.layers.Dense(self.flow_output_dims[l],
                                               activation=None,))
                                               #kernel_intializer=self.kernel_initializer))

  def call(self, input_tensor):
    flattened_out = self.flattened(input_tensor)
    e1_out = self.e1(flattened_out)
    e2_out = self.e2(e1_out)
    means_out = self.means(e2_out)
    log_var_out = self.log_var(e2_out)
    uv_out = []
    b_out = []
    for l in range(self.flow_num_hidden+1):
      u_out = tf.reshape(self.flow_U[l](e2_out),
                         (-1, self.flow_output_dims[l], self.flow_mat_rank))
      v_out = tf.reshape(self.flow_V[l](e2_out),
                         (-1, self.flow_mat_rank, self.flow_input_dims[l]))
      uv_out.append(tf.matmul(u_out, v_out))
      b_out.append(tf.reshape(self.flow_b[l](e2_out), (-1, self.flow_output_dims[l], 1)))
    return means_out, log_var_out, uv_out, b_out


class flow_net(tf.keras.layers.Layer):
  """Neural network that specifies the normalizing flow procedure. Can be used
for both a FFJORD flow or a RealNVP flow. The latter is accomodated by a split_nvp
flag that divides the output into two tensors of equal size for the scale and
translation operations.
  """

  def __init__(self, data_dim, name='flow_net', num_hidden=2, hidden_dim=200,
               kernel_initializer='gorot_normal', activation=tf.nn.tanh,
               nvp_split=False, **kwargs):
    super(flow_net, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.num_hidden = num_hidden
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.activation = activation
    self.nvp_split = nvp_split
    #If for realNVP layer, want to double output and split at end
    #This means realNVP scaling and translation computed by same neural net
    if self.nvp_split:
      self.output_dims = self.num_hidden*[self.hidden_dim] + [self.data_dim*2,]
    else:
      self.output_dims = self.num_hidden*[self.hidden_dim] + [self.data_dim,]
    #Create list of neural net layers
    self.layer_list = []
    for l in range(self.num_hidden+1):
      if l == self.num_hidden:
        this_activation=None
      else:
        this_activation = self.activation
      self.layer_list.append(tf.keras.layers.Dense(self.output_dims[l],
                                                   activation=this_activation,
                                                   name="d%i"%l,
                                                   kernel_initializer=self.kernel_initializer))

  def call(self, time, state=None):
    #Need to take an argument "time" so works with tfp.bijectors.FFJORD
    #Will not use this variable, though
    #So delete if we're using FFJORD
    #But if we're using RealNVP, the first argument is actually the state
    if isinstance(time, (int, float)) or len(time.shape) <= 1:
      del time
    else:
      #State will have to be a tensor or array and will have to have at least dimension 2
      #This is because the first dimension has to be batch size for tensorflow to work
      #Time, on the other hand, will only have one dimension or be an int or float
      state = time
    #Pass through layers to calculate output
    out = self.layer_list[0](state)
    for l in range(1, self.num_hidden+1):
      out = self.layer_list[l](out)
    if self.nvp_split:
      return tf.split(out, 2, axis=-1)
    else:
      return out


# class flow_net(tf.keras.layers.Layer):
#   """Neural network that specifies the integrand of the normalizing flow.
#      Similar to a dense neural net, but modified to have some parameters in each
#      layer depend on the initial encoding, as in Grathwohl, et al 2018 (FFJORD).
#      Also adapted to work as neural net to predict scale and translation in RealNVP,
#      but with modification proposed in FFJORD paper to have global layer weights
#      that are output by the encoder (if you want).
#   """
# 
#   def __init__(self, data_dim, name='flow_net', num_hidden=2, hidden_dim=200,
#                kernel_initializer='glorot_normal', activation='softplus',
#                nvp_split=False, **kwargs):
#     super(flow_net, self).__init__(name=name, **kwargs)
#     self.data_dim = data_dim
#     self.num_hidden = num_hidden
#     self.hidden_dim = hidden_dim
#     self.kernel_initializer = kernel_initializer
#     if activation == 'softplus':
#       self.activation = tf.nn.softplus
#     elif activation == 'relu':
#       self.activation = tf.nn.relu
#     elif activation == 'tanh':
#       self.activation = tf.nn.tanh
#     else:
#       print('Activation not recognized, setting to softplus')
#       self.activation = tf.nn.softplus
#     self.nvp_split = nvp_split
#     #Create layer weights
#     self.w = []
#     self.b = []
#     self.input_dims = [self.data_dim,] + self.num_hidden*[self.hidden_dim]
#     #If for realNVP layer, want to double output and split at end
#     #This means realNVP scaling and translation computed by same neural net
#     if self.nvp_split:
#       self.output_dims = self.num_hidden*[self.hidden_dim] + [self.data_dim*2,]
#     else:
#       self.output_dims = self.num_hidden*[self.hidden_dim] + [self.data_dim,]
#     for l in range(self.num_hidden+1):
#       self.w.append(self.add_weight(shape=(self.output_dims[l], self.input_dims[l]),
#                                     name='w_%i'%l,
#                                     initializer=self.kernel_initializer,
#                                     trainable=True))
#       self.b.append(self.add_weight(shape=(self.output_dims[l], 1),
#                                     name='b_%i'%l,
#                                     initializer=self.kernel_initializer,
#                                     trainable=True))
#     #And set up layer-based parameters that are specified by the encoder
#     #Instead of inputing to function, make adjustable and create function to set them
#     self.uv_list = [0]*(self.num_hidden+1)
#     self.b_list = [0]*(self.num_hidden+1)
# 
#   def set_uv_b(self, uv_list=None, b_list=None, flip_sign=False):
#     if uv_list is not None:
#       self.uv_list = copy.deepcopy(uv_list)
#       #How we do this depends on if using realNVP or not
#       if self.nvp_split:
#         if flip_sign:
#           self.uv_list[0] = self.uv_list[0][:, :, -self.data_dim:]
#           self.uv_list[-1] = self.uv_list[-1][:, -self.data_dim*2:, :]
#         else:
#           self.uv_list[0] = self.uv_list[0][:, :, :self.data_dim]
#           self.uv_list[-1] = self.uv_list[-1][:, :self.data_dim*2, :]
#     if b_list is not None:
#       self.b_list = copy.deepcopy(b_list)
#       if self.nvp_split:
#         if flip_sign:
#           self.b_list[-1] = self.b_list[-1][:, -self.data_dim*2:, :]
#         else:
#           self.b_list[-1] = self.b_list[-1][:, :self.data_dim*2, :]
# 
#   def call(self, time, state=None,
#            uv_list=None, b_list=None):
#     #Need to take an argument "time" so works with tfp.bijectors.FFJORD
#     #Will not use this variable, though
#     #So delete if we're using FFJORD
#     #But if we're using RealNVP, the first argument is actually the state
#     if isinstance(time, (int, float)) or len(time.shape) <= 1:
#       del time
#     else:
#       #State will have to be a tensor or array and will have to have at least dimension 2
#       #This is because the first dimension has to be batch size for tensorflow to work
#       #Time, on the other hand, will only have one dimension or be an int or float
#       state = time
#     #Can optionally set uv and b by passing to this function
#     self.set_uv_b(uv_list=uv_list, b_list=b_list)
# #    if uv_list is None:
# #      uv_list = [0]*(self.num_hidden+1)
# #    if b_list is None:
# #      b_list = [0]*(self.num_hidden+1)
#     #Pass through layers to calculate output
#     out = tf.reshape(state, state.shape+(1,))
#     for l in range(self.num_hidden+1):
#       out = tf.matmul(self.w[l] + self.uv_list[l], out)
#       out = out + self.b[l] + self.b_list[l]
# #      out = tf.matmul(self.w[l] + uv_list[l], out)
# #      out = out + self.b[l] + b_list[l]
#       if l < self.num_hidden:
#         out = self.activation(out)
#     out = tf.squeeze(out, axis=-1)
#     if self.nvp_split:
#       return tf.split(out, 2, axis=-1)
#     else:
#       return out


class NormFlowFFJORD(tf.keras.layers.Layer):
  """Normalizing flow layer using FFJORD.
  """

  def __init__(self, data_dim, name='ffjord_flow', flow_time=1.0,
               kernel_initializer='glorot_normal', flow_net_params={},
               **kwargs):
    super(NormFlowFFJORD, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.flow_time = flow_time
    self.kernel_initializer = kernel_initializer
    #Create the neural network that represents the flow kernel or integrand
    self.kernel = flow_net(self.data_dim,
                           kernel_initializer=self.kernel_initializer,
                           **flow_net_params)
    #And create the normalizing flow
    self.flow = tfp.bijectors.FFJORD(self.kernel,
                                     initial_time=0.0, final_time=self.flow_time,
                                     trace_augmentation_fn=tfp.bijectors.ffjord.trace_jacobian_exact)

  def call(self, input_tensor, reverse=False):
    if not reverse:
      #out = self.flow.forward(input_tensor)
      #log_det = self.flow.forward_log_det_jacobian(input_tensor,
      #                                             event_ndims=len(input_tensor.shape)-1)
      out, log_det = self.flow._augmented_forward(input_tensor)
    else:
      #out = self.flow.inverse(input_tensor)
      #log_det = self.flow.inverse_log_det_jacobian(input_tensor,
      #                                             event_ndims=len(input_tensor.shape)-1)
      out, log_det = self.flow._augmented_inverse(input_tensor)
    #If use augmented call, should take half the time, but have to sum over log_det
    log_det = tf.reduce_sum(log_det, axis=np.arange(1, len(input_tensor.shape)))
    return out, log_det


class NormFlowRealNVP(tf.keras.layers.Layer):
  """Normalizing flow layer using RealNVP.
  """

  def __init__(self, data_dim, name='realnvp_flow', num_blocks=8,
               kernel_initializer='truncated_normal', flow_net_params={},
               **kwargs):
    super(NormFlowRealNVP, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.num_blocks = num_blocks
    self.kernel_initializer = kernel_initializer
    #Want to create a neural network for the scale and shift transformations
    #(one for each desired number of blocks - num_blocks should be at least 2)
    #In case data_dim is not even, figure out lengths of split
    self.split_lens = np.zeros(self.num_blocks, dtype=int)
    if self.data_dim == 1:
      self.split_lens[:] = 1
    else:
      self.split_lens[::2] = self.data_dim//2
      self.split_lens[1::2] = self.data_dim - self.data_dim//2
    self.net_list = []
#    self.block_list = []
    #Make sure we're using RealNVP split
    flow_net_params['nvp_split'] = True
    for l in range(self.num_blocks):
      #Will use same neural network to predict S and T by transforming inputs
      #So nvp_split should be true for flow_net
      self.net_list.append(flow_net(self.split_lens[l],
                                    kernel_initializer=self.kernel_initializer,
                                    **flow_net_params))
#      this_n_masked = ((-1)**(l+1))*(self.data_dim - self.split_lens[l])
#      self.block_list.append(tfp.bijectors.RealNVP(num_masked=this_n_masked,
#                                                   shift_and_log_scale_fn=self.net_list[l]))

  def call(self, input_tensor, reverse=False):
    out = input_tensor
    log_det_sum = tf.zeros(input_tensor.shape[0])
#    if not reverse:
#      for block in self.block_list:
#        log_det_sum += block.forward_log_det_jacobian(out,
#                                                      event_ndims=len(input_tensor.shape)-1)
#        out = block.forward(out)
#    else:
#      for block in self.block_list:
#        log_det_sum += block.inverse_log_det_jacobian(out,
#                                                      event_ndims=len(input_tensor.shape)-1)
#        out = block.inverse(out)
    #If going backwards, reverse block order
    if not reverse:
      block_order = np.arange(self.num_blocks)
    else:
      block_order = np.arange(self.num_blocks-1, -1, -1)
    #Loop over blocks
    for l in block_order:
      this_n_masked = ((-1)**(l+1))*(self.data_dim - self.split_lens[l])
      if this_n_masked < 0:
        split0, split1 = out[:, this_n_masked:], out[:, :this_n_masked]
      else:
        split0, split1 = out[:, :this_n_masked], out[:, this_n_masked:]
      s_out, t_out = self.net_list[l](split0)
      if not reverse:
        transform_out = split1*tf.exp(s_out) + t_out
        log_det_sum += tf.reduce_sum(s_out, axis=1)
      else:
        transform_out = (split1 - t_out)*tf.exp(-s_out)
        log_det_sum -= tf.reduce_sum(s_out, axis=1)
      if this_n_masked < 0:
        out = tf.concat([transform_out, split0], axis=1)
      else:
        out = tf.concat([split0, transform_out], axis=1)
    return out, log_det_sum


# class NormFlowRealNVP(tf.keras.layers.Layer):
#   """Normalizing flow layer using RealNVP.
#   """
# 
#   def __init__(self, data_dim, name='realnvp_flow', num_blocks=8,
#                kernel_initializer='truncated_normal', flow_net_params={},
#                **kwargs):
#     super(NormFlowRealNVP, self).__init__(name=name, **kwargs)
#     self.data_dim = data_dim
#     #If data dimension is one, we will need to augment with ones
#     if self.data_dim == 1:
#       self.data_dim += 1
#     self.num_blocks = num_blocks
#     self.kernel_initializer = kernel_initializer
#     #Want to create a neural network for the scale and shift transformations
#     #(one for each desired number of blocks - num_blocks should be at least 2)
#     #In case data_dim is not even, figure out lengths of split
#     self.split_lens = np.zeros(self.num_blocks, dtype=int)
#     self.split_lens[::2] = self.data_dim//2
#     self.split_lens[1::2] = self.data_dim - self.data_dim//2
#     self.net_list = []
# #    self.block_list = []
#     #Make sure we're using RealNVP split
#     flow_net_params['nvp_split'] = True
#     for l in range(self.num_blocks):
#       #Will use same neural network to predict S and T by transforming inputs
#       #So nvp_split should be true for flow_net
#       self.net_list.append(flow_net(self.split_lens[l],
#                                     kernel_initializer=self.kernel_initializer,
#                                     **flow_net_params))
# #      this_n_masked = ((-1)**(l+1))*(self.data_dim - self.split_lens[l])
# #      self.block_list.append(tfp.bijectors.RealNVP(num_masked=this_n_masked,
# #                                                   shift_and_log_scale_fn=self.net_list[l]))
# 
#   def call(self, input_tensor, uv_list=None, b_list=None, reverse=False):
#     #To include inputs from encoder, need to update neural nets in all of our blocks
#     for i, net in enumerate(self.net_list):
#       net.set_uv_b(uv_list=uv_list, b_list=b_list, flip_sign=bool(i%2))
#     #If data dimension is 1, we need to augment with dummy data so transformation works
#     #Will throw away this added dimension/data at the end
#     if input_tensor.shape[1] == 1:
#       out = tf.concat((input_tensor, tf.ones(input_tensor.shape)), axis=1)
#     else:
#       out = input_tensor
#     log_det_sum = tf.zeros(input_tensor.shape[0])
#     if not reverse:
#       for l in range(self.num_blocks):
#         if l%2 == 0:
#           this_mask = out[:, -(self.data_dim - self.split_lens[l]):]
# #          this_uv = uv_list
# #          if uv_list is not None:
# #            this_uv[0] = uv_list[0][:, :, -(self.data_dim - self.split_lens[l]):]
# #            this_uv[-1] = uv_list[-1][:, -(self.data_dim - self.split_lens[l])*2:, :]
# #          this_b = b_list
# #          if b_list is not None:
# #            this_b[-1] = b_list[-1][:, -(self.data_dim - self.split_lens[l])*2:, :]
#           s_out, t_out = self.net_list[l](this_mask)#, uv_list=this_uv, b_list=this_b)
#           this_transform = out[:, :self.split_lens[l]]*tf.exp(s_out) + t_out
#           out = tf.concat([this_transform, this_mask], axis=1)
#         else:
#           this_mask = out[:, :(self.data_dim - self.split_lens[l])]
# #          this_uv = uv_list
# #          if uv_list is not None:
# #            this_uv[0] = uv_list[0][:, :, :(self.data_dim - self.split_lens[l])]
# #            this_uv[-1] = uv_list[-1][:, :(self.data_dim - self.split_lens[l])*2, :]
# #          this_b = b_list
# #          if b_list is not None:
# #            this_b[-1] = b_list[-1][:, :(self.data_dim - self.split_lens[l])*2, :]
#           s_out, t_out = self.net_list[l](this_mask)#, uv_list=this_uv, b_list=this_b)
#           this_transform = out[:, -self.split_lens[l]:]*tf.exp(s_out) + t_out
#           out = tf.concat([this_mask, this_transform], axis=1)
#         #If input shape is 1, only add to log tensor if fake data was masked
#         if input_tensor.shape[1] == 1:
#           if l%2 != 0:
#             log_det_sum += tf.reduce_sum(s_out, axis=1)
#         else:
#           log_det_sum += tf.reduce_sum(s_out, axis=1)
# #      for block in self.block_list:
# #        log_det_sum += block.forward_log_det_jacobian(out,
# #                                                      event_ndims=len(input_tensor.shape)-1)
# #        out = block.forward(out)
#     else:
#       #In reverse, go over transformation blocks backwards
#       for l in range(self.num_blocks-1, -1, -1):
#         if l%2 == 0:
#           this_mask = out[:, -(self.data_dim - self.split_lens[l]):]
# #          this_uv = uv_list
# #          if uv_list is not None:
# #            this_uv[0] = uv_list[0][:, :, -(self.data_dim - self.split_lens[l]):]
# #            this_uv[-1] = uv_list[-1][:, -(self.data_dim - self.split_lens[l])*2:, :]
# #          this_b = b_list
# #          if b_list is not None:
# #            this_b[-1] = b_list[-1][:, -(self.data_dim - self.split_lens[l])*2:, :]
#           s_out, t_out = self.net_list[l](this_mask)#, uv_list=this_uv, b_list=this_b)
#           this_transform = (out[:, :self.split_lens[l]] - t_out)*tf.exp(-s_out)
#           out = tf.concat([this_transform, this_mask], axis=1)
#         else:
#           this_mask = out[:, :(self.data_dim - self.split_lens[l])]
# #          this_uv = uv_list
# #          if uv_list is not None:
# #            this_uv[0] = uv_list[0][:, :, :(self.data_dim - self.split_lens[l])]
# #            this_uv[-1] = uv_list[-1][:, :(self.data_dim - self.split_lens[l])*2, :]
# #          this_b = b_list
# #          if b_list is not None:
# #            this_b[-1] = b_list[-1][:, :(self.data_dim - self.split_lens[l])*2, :]
#           s_out, t_out = self.net_list[l](this_mask)#, uv_list=this_uv, b_list=this_b)
#           this_transform = (out[:, -self.split_lens[l]:] - t_out)*tf.exp(-s_out)
#           out = tf.concat([this_mask, this_transform], axis=1)
#         #If input shape is 1, only add to log tensor if fake data was masked
#         if input_tensor.shape[1] == 1:
#           if l%2 != 0:
#             log_det_sum -= tf.reduce_sum(s_out, axis=1)
#         else:
#           log_det_sum -= tf.reduce_sum(s_out, axis=1)
# #      for block in self.block_list:
# #        log_det_sum += block.inverse_log_det_jacobian(out,
# #                                                      event_ndims=len(input_tensor.shape)-1)
# #        out = block.inverse(out)
#     if input_tensor.shape[1] == 1:
#       out = out[:, :-1]
#     return out, log_det_sum


class DiscriminatorNet(tf.keras.layers.Layer):
  """Discriminator network for use with Adversarial VAE (see Mescheder et al. 2017,
"Adversarial Variational Bayes..." for details. Note that two inputs are expected, one
for x and the second for z, the latent configuration.
  """

  def __init__(self, name='discriminator',
               hidden_dim_x = 1200,
               hidden_dim_z = 200,
               kernel_initializer='glorot_uniform',
               **kwargs):
    super(DiscriminatorNet, self).__init__(name=name, **kwargs)
    self.hidden_dim_x = hidden_dim_x
    self.hidden_dim_z = hidden_dim_z
    self.kernel_initializer = kernel_initializer
    self.flattened = tf.keras.layers.Flatten()
    self.dx1 = tf.keras.layers.Dense(self.hidden_dim_x, activation=tf.nn.relu, name="dx1",
                                     kernel_initializer=self.kernel_initializer)
    self.dx2 = tf.keras.layers.Dense(self.hidden_dim_x, activation=tf.nn.relu, name="dx2",
                                     kernel_initializer=self.kernel_initializer)
    self.dx3 = tf.keras.layers.Dense(self.hidden_dim_x, activation=None, name="dx3",
                                     kernel_initializer=self.kernel_initializer)
    self.dz1 = tf.keras.layers.Dense(self.hidden_dim_z, activation=tf.nn.relu, name="dz1",
                                     kernel_initializer=self.kernel_initializer)
    self.dz2 = tf.keras.layers.Dense(self.hidden_dim_z, activation=tf.nn.relu, name="dz2",
                                     kernel_initializer=self.kernel_initializer)
    #Last layer for z needs to have same shape as x
    self.dz3 = tf.keras.layers.Dense(self.hidden_dim_x, activation=None, name="dz3",
                                     kernel_initializer=self.kernel_initializer)

  def call(self, x_input, z_input):
    flat_x = self.flattened(x_input)
    dx1_out = self.dx1(flat_x)
    dx2_out = self.dx2(dx1_out)
    x_out = self.dx3(dx2_out)
    dz1_out = self.dz1(z_input)
    dz2_out = self.dz2(dz1_out)
    z_out = self.dz3(dz2_out)
    out = tf.reduce_sum(x_out*z_out, axis=1)
    return out


class AdversarialEncoder(tf.keras.layers.Layer):
  """Encoder for use with Adversarial VAE (Mescheder et al. 2017). Takes x configurations
and draws random noise from a standard normal distribution that is also input to
the encoder.
  """

  def __init__(self, num_latent, name='encoder',
               hidden_dim_x = 1200,
               hidden_dim_e = 200,
               kernel_initializer='glorot_uniform',
               **kwargs):
    super(AdversarialEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.hidden_dim_x = hidden_dim_x
    self.hidden_dim_e = hidden_dim_e
    self.kernel_initializer = kernel_initializer
    self.flattened = tf.keras.layers.Flatten()
    self.dx1 = tf.keras.layers.Dense(self.hidden_dim_x, activation=tf.nn.relu, name="dx1",
                                     kernel_initializer=self.kernel_initializer)
    self.dx2 = tf.keras.layers.Dense(self.hidden_dim_x, activation=tf.nn.relu, name="dx2",
                                     kernel_initializer=self.kernel_initializer)
    self.dx3 = tf.keras.layers.Dense(self.num_latent, activation=None, name="dx3",
                                     kernel_initializer=self.kernel_initializer)
    self.de1 = tf.keras.layers.Dense(self.hidden_dim_e, activation=tf.nn.relu, name="de1",
                                     kernel_initializer=self.kernel_initializer)
    self.de2 = tf.keras.layers.Dense(self.hidden_dim_e, activation=tf.nn.relu, name="de2",
                                     kernel_initializer=self.kernel_initializer)
    self.de3 = tf.keras.layers.Dense(self.num_latent, activation=None, name="de3",
                                     kernel_initializer=self.kernel_initializer)

  def call(self, x_input):
    flat_x = self.flattened(x_input)
    dx1_out = self.dx1(flat_x)
    dx2_out = self.dx2(dx1_out)
    x_out = self.dx3(dx2_out)
    #Sample random noise from standard normal
    e_input = tf.random.normal((x_input.shape[0], self.num_latent), 0, 1)
    de1_out = self.de1(e_input)
    de2_out = self.de2(de1_out)
    e_out = self.de3(de2_out)
    out = x_out*e_out + x_out + e_out
    return out


class DimerCGMapping(tf.keras.layers.Layer):
  """Deterministic mapping of full dimer coordinates with solvent to just the dimer pair
distance (the coarse-grained coordinate).
  """

  def __init__(self, name='encoder', **kwargs):
    super(DimerCGMapping, self).__init__(name=name, **kwargs)

  def call(self, x_input):
    #Assumes data is in format [x1, y1, x2, y2, ...] where the first two particles are dimer
    #Should be if flatten Nx2 array with default flatten params in numpy or tensorflow
    d = tf.sqrt(tf.square(x_input[:,2] - x_input[:,0])
                + tf.square(x_input[:,3] - x_input[:,1]))
    return tf.reshape(d, d.shape+(1,))


