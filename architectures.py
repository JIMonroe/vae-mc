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
               flow_net_params={'num_hidden':2, 'hidden_dim':200},
               flow_mat_rank=1, **kwargs):
    super(FCEncoderFlow, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.flow_num_hidden = flow_net_params['num_hidden']
    self.flow_hidden_dim = flow_net_params['hidden_dim']
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
  """Neural network that specifies the integrand of the normalizing flow.
     Similar to a dense neural net, but modified to have some parameters in each
     layer depend on the initial encoding, as in Grathwohl, et al 2018 (FFJORD).
  """

  def __init__(self, data_dim, name='flow_net', num_hidden=2, hidden_dim=200,
               kernel_initializer='glorot_normal', activation='softplus', **kwargs):
    super(flow_net, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.num_hidden = num_hidden
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    if activation == 'softplus':
      self.activation = tf.nn.softplus
    elif activation == 'relu':
      self.activation = tf.nn.relu
    elif activation == 'tanh':
      self.activation = tf.nn.tanh
    else:
      print('Activation not recognized, setting to softplus')
      self.activation = tf.nn.softplus
    #Create layer weights
    self.w = []
    self.b = []
    self.input_dims = [self.data_dim,] + self.num_hidden*[self.hidden_dim]
    self.output_dims = self.num_hidden*[self.hidden_dim] + [self.data_dim,]
    for l in range(self.num_hidden+1):
      self.w.append(self.add_weight(shape=(self.output_dims[l], self.input_dims[l]),
                    initializer=self.kernel_initializer,
                    trainable=True))
      self.b.append(self.add_weight(shape=(self.output_dims[l], 1),
                    initializer=self.kernel_initializer,
                    trainable=True))
    #And set up layer-based parameters that are specified by the encoder
    #Instead of inputing to function, make adjustable and create function to set them
    self.uv_list = [0]*(self.num_hidden+1)
    self.b_list = [0]*(self.num_hidden+1)

  def set_uv_b(self, uv_list=None, b_list=None):
    if uv_list is not None:
      self.uv_list = uv_list
    if b_list is not None:
      self.b_list = b_list

  def call(self, time, state,
           uv_list=None, b_list=None):
    #Need to take an argument "time" so works with tfp.bijectors.FFJORD
    #Will not use this variable, though
    del time
    #Can optionally set uv and b by passing to this function
    self.set_uv_b(uv_list=uv_list, b_list=b_list)
    #Pass through layers to calculate output
    out = tf.reshape(state, state.shape+(1,))
    for l in range(self.num_hidden+1):
      out = tf.matmul(self.w[l] + self.uv_list[l], out)
      out = out + self.b[l] + self.b_list[l]
      if l < self.num_hidden:
        out = self.activation(out)
    return tf.squeeze(out, axis=-1)


class NormFlow(tf.keras.layers.Layer):
  """Normalizing flow layer
  """

  def __init__(self, data_dim, name='flow_net', flow_time=1.0,
               kernel_initializer='glorot_normal', flow_net_params={},
               **kwargs):
    super(NormFlow, self).__init__(name=name, **kwargs)
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

  def call(self, input_tensor, uv_list=None, b_list=None, reverse=False):
    #To include inputs from encoder, need to update them in our kernel
    self.kernel.set_uv_b(uv_list=uv_list, b_list=b_list)
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


