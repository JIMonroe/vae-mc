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

"""Library of neural network architectures used for constructing VAEs
Adapted from disentanglement_lib https://github.com/google-research/disentanglement_lib"""
import copy
import numpy as np
import scipy.interpolate as si
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

  Modified to account for periodic degrees of freedom (convert to sine-cosine pairs)
  if there are any specified.

  Returns:
    means: Output tensor of shape (batch_size, num_latent) with latent variable
      means.
    log_var: Output tensor of shape (batch_size, num_latent) with latent
      variable log variances.
  """

  def __init__(self, num_latent, name='encoder', hidden_dim=1200,
               periodic_dofs=[False,],
               kernel_initializer='glorot_uniform',
               **kwargs):
    super(FCEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.hidden_dim = hidden_dim
    #Set up periodic DOF boolean list
    self.any_periodic = np.any(periodic_dofs)
    self.periodic_dofs = periodic_dofs
    self.kernel_initializer = kernel_initializer
    self.flattened = tf.keras.layers.Flatten()
    self.e1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e1",
                                    kernel_initializer=self.kernel_initializer)
    self.e2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name="e2",
                                    kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(self.num_latent, activation=None,
                                       kernel_initializer=self.kernel_initializer)
    self.log_var = tf.keras.layers.Dense(self.num_latent, activation=None,
                                         kernel_initializer=self.kernel_initializer)

  def call(self, input_tensor):
    flattened_out = self.flattened(input_tensor)
    #If have periodic DOFs, want to convert to 2D non-periodic coordinates in first step
    if self.any_periodic:
      flattened_out_p = tf.boolean_mask(flattened_out, self.periodic_dofs, axis=1)
      flattened_out_nonp = tf.boolean_mask(flattened_out,
                                           tf.math.logical_not(self.periodic_dofs), axis=1)
      cos_p = tf.math.cos(flattened_out_p)
      sin_p = tf.math.sin(flattened_out_p)
      flattened_out = tf.concat([flattened_out_nonp, cos_p, sin_p], axis=-1)
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
                                     padding="valid",
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
  """Fully connected decoder
  """

  def __init__(self, out_shape, name='decoder',
               hidden_dim=1200,
               kernel_initializer='glorot_uniform',
               return_vars=False,
               **kwargs):
    super(FCDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.hidden_dim = hidden_dim
    self.kernel_initializer=kernel_initializer
    self.return_vars = return_vars
    self.d1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    self.d2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    self.means = tf.keras.layers.Dense(np.prod(self.out_shape), activation=None,
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
    #Based on output shape, figure out best starting convolution
    #Each convolutional layer has stride of 2, so reduces by roughly factor of 2 each time
    #Have 4 convolutional layers, so subtract from the power that gets you to the output
    #Only works if square image
    self.out_shape = out_shape
    self.conv_start_shape = int(2**(np.ceil(np.log(self.out_shape[0])/np.log(2.0)) - 4))
    self.kernel_initializer = kernel_initializer
    self.d1 = tf.keras.layers.Dense(256, activation=tf.nn.relu,
                                    kernel_initializer=self.kernel_initializer)
    self.d2 = tf.keras.layers.Dense(self.conv_start_shape*self.conv_start_shape*64,
                                    activation=tf.nn.relu,
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
    self.d6 = tf.keras.layers.Conv2DTranspose(filters=self.out_shape[2],
                                              kernel_size=4,
                                              strides=2,
                                              padding="valid",
                                              kernel_initializer=self.kernel_initializer,
                                             )

  def call(self, latent_tensor):
    d1_out = self.d1(latent_tensor)
    d2_out = self.d2(d1_out)
    d2_reshaped = tf.reshape(d2_out, shape=[-1, self.conv_start_shape,
                                            self.conv_start_shape, 64])
    d3_out = self.d3(d2_reshaped)
    d4_out = self.d4(d3_out)
    d5_out = self.d5(d4_out)
    d6_out = self.d6(d5_out)
    #return tf.reshape(d6_out, (-1,) + self.out_shape)
    #Will now have too many output values, so throw away extras
    #Not exactly elegant, but allows to easily change image shape
    return d6_out[:, :self.out_shape[0], :self.out_shape[1], :]


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
#      for block in self.block_list[::-1]:
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


class SplineBijector(tf.keras.layers.Layer):
  """Follows tfp example for using rational quadratic splines (as described in Durkan et al.
2019) to replace affine transformations (in, say, RealNVP). This should allow more flexible
transformations with similar cost and should work much better with 1D flows.
  """

  def _bin_positions(self, x):
    x = tf.reshape(x, [tf.shape(x)[0], -1, self.num_bins])
    out = tf.math.softmax(x, axis=-1)
    out = out*(self.bin_max - self.bin_min - self.num_bins*1e-2) + 1e-2
    return out

  def _slopes(self, x):
    x = tf.reshape(x, [tf.shape(x)[0], -1, self.num_bins - 1])
    return tf.math.softplus(x) + 1e-2

  def __init__(self, data_dim, name='rqs',
               bin_range=[-10.0, 10.0],
               num_bins=32, hidden_dim=200,
               kernel_initializer='truncated_normal',
               **kwargs):
    super(SplineBijector, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.bin_min = bin_range[0]
    self.bin_max = bin_range[1]
    self.num_bins = num_bins
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    #Create an initial neural net layer
    self.d1 = tf.keras.layers.Dense(self.hidden_dim, name='d1',
                                    activation=tf.nn.relu,
                                    kernel_initializer=self.kernel_initializer)
    #Create neural nets for widths, heights, and slopes
    self.bin_widths = tf.keras.layers.Dense(self.data_dim*self.num_bins,
                                            activation=self._bin_positions,
                                            name='w',
                                            kernel_initializer=self.kernel_initializer)
    self.bin_heights = tf.keras.layers.Dense(self.data_dim*self.num_bins,
                                             activation=self._bin_positions,
                                             name='h',
                                             kernel_initializer=self.kernel_initializer)
    self.knot_slopes = tf.keras.layers.Dense(self.data_dim*(self.num_bins - 1),
                                             activation=self._slopes,
                                             name='s',
                                             kernel_initializer=self.kernel_initializer)

  def call(self, input_tensor, nunits):
    #Don't use nunits because more efficient to create nets beforehand
    del nunits
    if input_tensor.shape[1] == 0:
      input_tensor = tf.ones((input_tensor.shape[0], 1))
    d1_out = self.d1(input_tensor)
    return tfp.bijectors.RationalQuadraticSpline(bin_widths=self.bin_widths(d1_out),
                                                 bin_heights=self.bin_heights(d1_out),
                                                 knot_slopes=self.knot_slopes(d1_out),
                                                 range_min=self.bin_min)


class NormFlowRQSplineRealNVP(tf.keras.layers.Layer):
  """Follows tfp example for using rational quadratic splines (as described in Durkan et al.
2019) with the RealNVP structure. This should allow more flexible transformations with
similar cost and  should work much better for 1D flows.
  """

  def __init__(self, data_dim, name='rqs_realnvp_flow', num_blocks=4,
               kernel_initializer='truncated_normal', rqs_params={},
               **kwargs):
    super(NormFlowRQSplineRealNVP, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.num_blocks = num_blocks
    self.kernel_initializer = kernel_initializer
    self.rqs_params = rqs_params
    #Want to create a spline bijector for each block
    #(one for each desired number of blocks - num_blocks should be at least 2)
    #In case data_dim is not even, figure out lengths of split
    self.split_lens = np.zeros(self.num_blocks, dtype=int)
    if self.data_dim == 1:
      self.split_lens[:] = 1
    else:
      self.split_lens[::2] = self.data_dim//2
      self.split_lens[1::2] = self.data_dim - self.data_dim//2
    self.net_list = []
    self.block_list = []
    for l in range(self.num_blocks):
      self.net_list.append(SplineBijector(self.split_lens[l],
                                          kernel_initializer=self.kernel_initializer,
                                          **rqs_params))
      this_n_masked = ((-1)**(l+1))*(self.data_dim - self.split_lens[l])
      self.block_list.append(tfp.bijectors.RealNVP(num_masked=this_n_masked,
                                                   name='block_%i'%l,
                                                   bijector_fn=self.net_list[l]))

  def call(self, input_tensor, reverse=False):
    out = input_tensor
    log_det_sum = tf.zeros(tf.shape(input_tensor)[0])
    if not reverse:
      for block in self.block_list:
        log_det_sum += block.forward_log_det_jacobian(out,
                                                      event_ndims=len(input_tensor.shape)-1)
        out = block.forward(out)
    else:
      for block in self.block_list[::-1]:
        log_det_sum += block.inverse_log_det_jacobian(out,
                                                      event_ndims=len(input_tensor.shape)-1)
        out = block.inverse(out)
    return out, log_det_sum


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


class LatticeGasCGMapping(tf.keras.layers.Layer):
  """Deterministic mapping of full LG configuration to average density per site.
  """

  def __init__(self, name='encoder', **kwargs):
    super(LatticeGasCGMapping, self).__init__(name=name, **kwargs)

  def call(self, x_input):
    n = tf.reduce_sum(x_input, axis=np.arange(1, len(x_input.shape)))
    dens = n / tf.cast(tf.reduce_prod(x_input.shape[1:]), 'float32')
    return tf.reshape(dens, dens.shape+(1,))


class SplinePotential(tf.keras.layers.Layer):
  """Defines a potential energy in terms of splines for use with relative entropy coarse-
graining.
  """

  def __init__(self, name='spline_u', knot_points=None, beta=1.0, **kwargs):
    super(SplinePotential, self).__init__(name=name, **kwargs)
    if knot_points is None:
      self.knots = np.linspace(0.0, 1.0, 50)
    else:
      self.knots = knot_points
    self.beta = beta
    #Set up coefficients for 3rd order (cubic) splines
    self.coeffs = self.add_weight(shape=(self.knots.shape[0] - 4,), name='coeffs',
                                  initializer='ones', trainable=True)
    self.bspline = si.BSpline(self.knots,
                              self.coeffs.numpy().astype('float64'),
                              3, extrapolate=True)

  def call(self, x):
    """For the given positions, returns the spline values at those positions. Outside of
the spline range, extrapolation is used (see scipy.interpolate.BSpline).
    """
    #Always make sure using latest coefficients
    self.bspline.c = self.coeffs.numpy().astype('float64')
    return self.beta*self.bspline(x.numpy())

  def get_coeff_derivs(self, x):
    """Returns the derivatives with respect to all coefficients (gradient vector) for all of
the input positions. The result is NxM where N is the number of input positions and M is the
number of coefficients.
    """
    #Always make sure using latest coefficients
    self.bspline.c = self.coeffs.numpy().astype('float64')
    #Obtain derivatives by looping over setting difference coefficients to zero
    coeff_derivs = np.zeros((x.shape[0], self.coeffs.shape[0]))
    temp_coeffs = np.eye(self.coeffs.shape[0])
    for i, cvec in enumerate(temp_coeffs):
      self.bspline.c = cvec
      coeff_derivs[:, i] = self.bspline(x)
    self.bspline.c = self.coeffs.numpy().astype('float64')
    return self.beta*coeff_derivs


class LatticeGasCGReduceMap(tf.keras.layers.Layer):
  """Deterministic OR stochastic mapping of lattice gas to smaller lattice. Specify the
number of sites to group as n_group, which should divide the larger lattice into an integer
number of new sites. For deterministic maps, have two options: average the sites and assign
this non-integer value, or average the sites and assign 0 if < 0.5 or 1 if > 0.5. Can also
look at latter as type of deterministic "sampling," so use sample_stochastic to switch
between this and drawing from a Bernoulli distribution based on the average.
  """

  def __init__(self, name='encoder',
               n_group= 4,
               sample=True, sample_stochastic=True,
               **kwargs):
    super(LatticeGasCGReduceMap, self).__init__(name=name, **kwargs)
    self.n_group = n_group
    self.sample = sample
    self.sample_stochastic = sample_stochastic

  def call(self, x_input):
    #Do pooled averaging over sites
    #If not an integer, pads as evenly as possible on both sides
    #Padded values do not contribute to the average, though
    pooled_x = tf.nn.avg_pool(x_input, (self.n_group, self.n_group),
                              (self.n_group, self.n_group), 'SAME')
    if self.sample:
      if self.sample_stochastic:
        rand_vals = tf.random.uniform(pooled_x.shape)
      else:
        rand_vals = 0.5
      out = tf.cast((pooled_x > rand_vals), dtype='float32')
    else:
      out = pooled_x
    return out


class ReducedLGPotential(tf.keras.layers.Layer):
  """Reproduces the lattice gas Hamiltonian in losses, but makes mu and epsilon
adjustable tensorflow variables.
  """

  def __init__(self, name='lg_u', beta=1.0, **kwargs):
    super(ReducedLGPotential, self).__init__(name=name, **kwargs)
    #To make work more easily with spline-based code, have single vector of parameters
    #Less readable, but stating here that first position is mu, second is epsilon
    self.coeffs = self.add_weight(name='mu_eps', shape=(2,), trainable=True)
    self.beta = beta

  def call(self, x):
    #Shift all indices by 1 in up and down then left and right and multiply by original
    ud = self.coeffs[1]*x*tf.roll(x, 1, axis=1)
    lr = self.coeffs[1]*x*tf.roll(x, 1, axis=2)
    #Next incorporate chemical potential term
    chempot = self.coeffs[0]*x
    #And sum everything up
    H = tf.reduce_sum(ud + lr - chempot, axis=np.arange(1, len(x.shape)))
    return self.beta*H

  def get_coeff_derivs(self, x):
    """Hard-coding of derivatives for Srel optimization. Note that returns derivative for
each input point. Tensorflow could take its own derivatives, but following what was done
for spline potential, where tensorflow cannot compute derivatives itself.
    """
    mu_deriv = -tf.reduce_sum(x, axis=np.arange(1, len(x.shape)))
    eps_deriv = tf.reduce_sum(x*tf.roll(x, 1, axis=1) + x*tf.roll(x, 1, axis=2),
                              axis=np.arange(1, len(x.shape)))
    return tf.stack([mu_deriv, eps_deriv], axis=1).numpy()


class MaskedNet(tf.keras.layers.Layer):
  """Dense neural network with connections controlled by a mask, which allows for creation
of autoregressive networks if the mask is properly designed. This is based on the code for
tfp.bijectors.AutoregressiveNetwork, but with more of a TF 2 feel, fewer features, but more
flexibility with the masking operation.
  """
  def __init__(self, num_params, event_shape=None,
               conditional=False, conditional_event_shape=None,
               hidden_units=[],
               dof_order=None,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(MaskedNet, self).__init__(**kwargs)
    self.num_params = num_params
    self.event_shape = event_shape
    self.conditional = conditional
    self.conditional_event_shape = conditional_event_shape
    if self.conditional and self.conditional_event_shape is None:
      raise ValueError('If want conditional input, must specify conditional_event_shape')
    self.hidden_units = hidden_units
    self.activation = tf.keras.layers.Activation(activation) #Will call directly
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    #Set up default ordering of degrees of freedom for masking if not specified
    if dof_order is None:
      #By default, just make the mask standardly autoregressive
      #This is equivalent to numbering the input degrees of freedom sequentially
      self.dof_order = np.arange(1, self.event_shape+1)
    #Otherwise just make sure it's a numpy array (for now... will be list of arrays)
    else:
      self.dof_order = np.array(dof_order)

    #To make sure input is 1D (other than batch dimension) will want to flatten
    self.flatten = tf.keras.layers.Flatten()

    #Will be created in build method
    self.masks = []
    self.networks = []
    self.conditional_networks = []

  @staticmethod
  def _make_masked_constraint(mask, constraint=None):
    constraint = tf.keras.constraints.get(constraint)
    def masked_constraint(w):
      if constraint is not None:
        w = constraint(w)
      return tf.cast(mask, w.dtype) * w
    return masked_constraint

  @staticmethod
  def _make_masked_initializer(mask, initializer):
    initializer = tf.keras.initializers.get(initializer)
    def masked_initializer(shape, dtype=None):
      w = initializer(shape, dtype)
      return tf.cast(mask, w.dtype) * w
    return masked_initializer

  def build(self, input_shape):
    if self.event_shape is None:
      self.event_shape = np.prod(input_shape[1:]) #Product because will flatten

    if np.prod(input_shape[1:]) != self.event_shape:
      raise ValueError('Specified event_shape and shape of provided input must match!')

    #For each hidden dimension, create a mask and a network
    layer_outputs = self.hidden_units + [self.event_shape*self.num_params,]
    self.dof_order = [self.dof_order,]
    max_degree = np.max(self.dof_order[-1]) - 1
    for k in range(len(layer_outputs)):
      if k != len(layer_outputs) - 1:
        next_dof_order = np.ceil(np.arange(1, layer_outputs[k] + 1) * (self.event_shape - 1)
                                 / float(layer_outputs[k] + 1)).astype(np.int32)
        next_dof_order = np.minimum(next_dof_order, max_degree)
        self.masks.append(self.dof_order[-1][:, np.newaxis] <= next_dof_order)
      else:
        next_dof_order = self.dof_order[0]
        self.masks.append(self.dof_order[-1][:, np.newaxis] < next_dof_order)
        self.masks[k] = np.reshape(np.tile(self.masks[k][..., np.newaxis],
                                           [1, 1, self.num_params]),
                                   [self.masks[k].shape[0], self.event_shape*self.num_params])
      self.dof_order.append(next_dof_order)
      self.networks.append(
                        tf.keras.layers.Dense(layer_outputs[k],
                        activation=None,
                        use_bias=self.use_bias,
                        kernel_initializer=self._make_masked_initializer(self.masks[k],
                                                                    self.kernel_initializer),
                        kernel_constraint=self._make_masked_constraint(self.masks[k], None)))
      if self.conditional:
        self.conditional_networks.append(tf.keras.layers.Dense(layer_outputs[k],
                                                 activation=None,
                                                 use_bias=False,
                                                 kernel_initializer=self.kernel_initializer))
    super(MaskedNet, self).build(input_shape)

  def call(self, input_tensor, conditional_input=None):
    out = self.flatten(input_tensor)
    for k in range(len(self.networks)):
      out = self.networks[k](out)
      if self.conditional:
        out = out + self.conditional_networks[k](conditional_input)
      if k != len(self.networks) - 1:
        out = self.activation(out)
    return tf.reshape(out, (-1, self.event_shape, self.num_params))


class AutoregressiveDecoder(tf.keras.layers.Layer):
  """Fully connected encoder with autoregressive network right before prediction.

  Args:
    latent_tensor: Input tensor to connect decoder to.
    out_shape: Shape of the data.
  """

  def __init__(self, out_shape, name='decoder',
               hidden_dim=1200,
               kernel_initializer='glorot_uniform',
               return_vars=False,
               skip_connections=True,
               auto_group_size=1,
               truncate_normal=False,
               periodic_dofs=[False,],
               **kwargs):
    super(AutoregressiveDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.hidden_dim = hidden_dim
    self.kernel_initializer=kernel_initializer
    self.return_vars = return_vars
    self.skips = skip_connections
    #Specify if normal distributions should be truncated at box size
    #ONLY APPLICABLE TO 2D PARTICLE DIMER SYSTEM!
    self.truncate_normal = truncate_normal
    #Specify which DOFs should be periodic and use VonMises distribution
    self.any_periodic = np.any(periodic_dofs)
    if self.any_periodic:
      #Check if number of specified DOFs match output shape
      if np.prod(out_shape) != len(periodic_dofs):
        raise ValueError('Length of periodic_dofs does not match flattened out_shape.'
                         +'\nIf specifying any periodic DOFs, must specify for all DOFs')
      else:
        self.periodic_dofs = periodic_dofs
    else:
      self.periodic_dofs = [False,]*np.prod(out_shape)
    #Specify how to group DOFs together when autoregressing based on group size
    self.auto_group_size = auto_group_size
    #For all separate (default if None when pass to MaskedNet) get [1, 2, 3, 4,...]
    #If wanted to group two DOFs at a time, would want [1, 1, 2, 2, 3, 3,...]
    self.auto_groupings = np.repeat(
                          np.arange(1, np.ceil(np.prod(out_shape)/self.auto_group_size) + 1),
                          self.auto_group_size)[:int(np.prod(out_shape))].astype(np.int32)
    self.d1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    self.d2 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    if self.return_vars:
      self.out_event_dims = 2
      if self.any_periodic:
        self.out_event_dims += 1
    else:
      self.out_event_dims = 1
    #Will predict means (and log variances) with neural net
    #Autoregressive network will then shift means and scale variances (so shift log vars)
    #The shifts will augment the base distribution as information on sampled configs is added
    self.base_param = tf.keras.layers.Dense(self.out_event_dims*np.prod(self.out_shape),
                                            activation=None,
                                            kernel_initializer=self.kernel_initializer)
    #And will need function to flatten training data if have it
    self.flatten = tf.keras.layers.Flatten()

  def build(self, input_shape):
    #If want conditional input, need to know latent dimension
    #So create in build method
    #With/without conditional input as latent space tensor if self.skips is true/false
    if self.skips:
#      self.autonet = tfp.bijectors.AutoregressiveNetwork(self.out_event_dims,
#                                                  event_shape=np.prod(self.out_shape),
#                                                  conditional=True,
#                                                  conditional_event_shape=input_shape[1:],
#                                                  conditional_input_layers='all_layers',
#                                                  hidden_units=[self.hidden_dim,],
#                                                  input_order='left-to-right',
#                                                  hidden_degrees='equal',
#                                                  activation=tf.nn.tanh,
#                                                  kernel_initializer=self.kernel_initializer)
      self.autonet = MaskedNet(self.out_event_dims,
                               event_shape=np.prod(self.out_shape),
                               conditional=True,
                               conditional_event_shape=input_shape[1:],
                               hidden_units=[self.hidden_dim,],
                               dof_order=self.auto_groupings,
                               activation='tanh',
                               use_bias=True,
                               kernel_initializer=self.kernel_initializer)
    else:
#      self.autonet = tfp.bijectors.AutoregressiveNetwork(self.out_event_dims,
#                                                  event_shape=np.prod(self.out_shape),
#                                                  hidden_units=[self.hidden_dim,],
#                                                  input_order='left-to-right',
#                                                  hidden_degrees='equal',
#                                                  activation=tf.nn.tanh,
#                                                  kernel_initializer=self.kernel_initializer)
      self.autonet = MaskedNet(self.out_event_dims,
                               event_shape=np.prod(self.out_shape),
                               hidden_units=[self.hidden_dim,],
                               dof_order=self.auto_groupings,
                               activation='tanh',
                               use_bias=True,
                               kernel_initializer=self.kernel_initializer)

  def _split_shift(self, orig_mean, orig_logvar, shift):
    """Two ways to split autoregressive network output depending on periodic DOFs.
       Assumes self.return_vars is True.
    """
    if self.any_periodic:
      cos_shift, sin_shift, logvar_shift = tf.squeeze(tf.split(shift, 3, axis=-1), axis=-1)
      #Should have list of 2 tensors representing cosine and sine pairs (for some DOFs)
      cos_out = orig_mean[0] + cos_shift
      sin_out = orig_mean[1] + sin_shift
      #For periodic DOFs, pass through arctan2
      mean_out_p = tf.math.atan2(sin_out, cos_out)
      #For non-periodic, just add together
      mean_out_nonp = sin_out + cos_out
      #Take only the elements we want from each (wastes comp, but differentiable and works)
      mean_out = tf.where(self.periodic_dofs, mean_out_p, mean_out_nonp)
      logvar_out = orig_logvar + logvar_shift
    else:
      mean_shift, logvar_shift = tf.squeeze(tf.split(shift, 2, axis=-1), axis=-1)
      mean_out = orig_mean + mean_shift
      logvar_out = orig_logvar + logvar_shift
    return mean_out, logvar_out

  def get_truncation(self, means, coords,
                     hs_diam=1.1611*0.82, #Actual hard-sphere diameter multiplied by 0.82
                                          #Prevents overlaps causing energy over 5.65 kB*T
                     box_dims=np.array([[-3.0, 3.0], [-3.0, 3.0]]),
                     num_skip=2, #Specific to dimer particles
                     ):
    """Given the means of Gaussian distributions for each coordinate and coordinate values,
uses an autoregressive-type model to truncate the distribution. For each mean, coordinates
of lower index are used to determine the low and high cutoffs for truncating its
distribution. These cutoffs associated with each mean are returned. If no other particles are
closer than the box edge locations specified by "box_dims" then the box edges are used for
truncation. Note that truncation will only be performed based on the same cartesian dimension
of previously generated particles with similar cartesian coordinates in the other dimensions.
    """
    if box_dims.shape[0] != self.auto_group_size:
      raise ValueError('Number of box dimensions for Gaussian truncation must match number of autoregressive groups! (Otherwise model will not be autoregressive)')
    #Set up upper and lower cutoffs, just using box-based cutoffs as defaults
    low_cuts = np.zeros(means.shape, dtype=np.float32)
    high_cuts = np.zeros(means.shape, dtype=np.float32)
    for i in range(self.auto_group_size):
      #Subtract or add hard-sphere radius to box lower or upper box locations
      low_cuts[:, i::self.auto_group_size] = box_dims[i, 0] - hs_diam*0.5
      high_cuts[:, i::self.auto_group_size] = box_dims[i, 1] + hs_diam*0.5
#    #Loop over means, ignoring first num_skip particles
#    for i in range(num_skip*self.auto_group_size, means.shape[-1], self.auto_group_size):
#      #Loop over dimensions to truncate in each one
#      for j in range(self.auto_group_size):
#        #Create boolean for eligible particles
#        this_eligible = np.zeros(means.shape, dtype=bool)
#        this_eligible[:, j:i:self.auto_group_size] = True
#        #Loop over dimensions other than this one to find particles this one would hit
#        #(if moved only along the j dimension)
#        #Conservatively using half the hard-sphere diameter to define if will hit
#        for k in range(self.auto_group_size):
#          if k == j:
#            continue
#          this_compare = tf.math.logical_and((coords[:, k:i:self.auto_group_size]
#                                              >= means[:, i+k:i+k+1]-0.5*hs_diam),
#                                             (coords[:, k:i:self.auto_group_size]
#                                              <= means[:, i+k:i+k+1]+0.5*hs_diam))
#          this_eligible[:, j:i:self.auto_group_size] *= this_compare.numpy()
#        low_eligible = this_eligible*(coords <= means[:, i+j:i+j+1]).numpy()
#        #For lower pick largest sampled coordinate below mean (above guarantees this)
#        low_cuts[:, i+j] = tf.reduce_max(tf.ragged.boolean_mask(coords+hs_diam,
#                                                                low_eligible), axis=-1)
#        #If box is closer, pick that instead
#        low_cuts[:, i+j] = tf.maximum(low_cuts[:, i+j], box_dims[j, 0] - hs_diam*0.5)
#        #Same for higher cutoff, but pick smallest sampled coordinate above mean
#        high_eligible = this_eligible*(coords > means[:, i+j:i+j+1]).numpy()
#        high_cuts[:, i+j] = tf.reduce_min(tf.ragged.boolean_mask(coords-hs_diam,
#                                                                 high_eligible), axis=-1)
#        high_cuts[:, i+j] = tf.minimum(high_cuts[:, i+j], box_dims[j, 1] + hs_diam*0.5)
#        #Finally need to make sure that low_cuts < high_cuts
#        #If this happens, just use the box cutoffs and truncation will not help the model
#        flipped_cuts = high_cuts[:, i+j] <= low_cuts[:, i+j]
#        low_cuts[flipped_cuts, i+j] = box_dims[j, 0] - hs_diam*0.5
#        high_cuts[flipped_cuts, i+j] = box_dims[j, 1] + hs_diam*0.5
    return low_cuts, high_cuts

  def create_dist(self, params):
    """Need a function to create a sampling distribution for the autoregressive distribution.
Will use Gaussian if return_vars is True and Bernoulli if False. Note that during training
no sampling is performed, only calculation of probabilities, allowing gradients to work.
Really just want to pass parameters - can use autoregressive network outside of this.
If return_vars is true, params should be a list of [means, logvars].
    """
    if self.return_vars:
      means = params[0]
      logvars = params[1]
      try:
        lowcuts = params[2]
        highcuts = params[3]
        base_dist = tfp.distributions.TruncatedNormal(means, tf.exp(0.5*logvars),
                                                      lowcuts, highcuts)
      except IndexError:
        #For periodic DOFs, need to build list of distributions
        #For 'True' values, use VonMises, for 'False' use Normal
        if self.any_periodic:
          base_dist_list = []
          for i in range(means.shape[1]):
            if self.periodic_dofs[i]:
              #With VonMises in tfp, will get NaN if logvars too negative
              #Issues with sampling if less than about -43 and issues with logP if about -80
              #Doesn't give infinity, though, just NaN due to sampling algorithm
              #So we will make sure it's bigger than -40
              #Well, those numbers work on a CPU... for a GPU you get random NaNs
              #starting at about -17 for the purposes of sampling
              base_dist_list.append(tfp.distributions.VonMises(means[:, i],
                                                  tf.exp(-tf.maximum(logvars[:, i], -15.0))))
              #Decided that if using periodic DOFs, should use throughout.
              #In other words, if periodic_dofs set to True for a VAE model, it means
              #that the encoder converts these to sine-cosine pairs and the decoder
              #produces outputs according to a periodic von Mises distribution.
              #The below instead assumes that you're given sine-cosine pairs and you
              #also output sine-cosine pairs, but you truncate the distribution.
              #This is not strictly necessary (because np.arctan2 will normalize) and in
              #simple tests with the polymer proved no better and maybe detrimental.
              #base_dist_list.append(tfp.distributions.TruncatedNormal(means[:, i],
              #                                                tf.exp(0.5*logvars[:, i]),
              #                                                -tf.ones((means.shape[0], 1)),
              #                                                tf.ones((means.shape[0], 1))))
            else:
              base_dist_list.append(tfp.distributions.Normal(means[:, i],
                                                             tf.exp(0.5*logvars[:, i])))
          #Join together into a joint distribution
          #This is not as nice to work with as Independent, but will be ok
          #log_prob() function will return in same way
          #But sample() will give list of tensors for each DOF, so will need to stack them
          #(and unstack to feed into log_prob())
          base_dist = tfp.distributions.JointDistributionSequential(base_dist_list)
        else:
          base_dist = tfp.distributions.Normal(means, tf.exp(0.5*logvars))
          #base_dist = tfp.distributions.Laplace(means, tf.exp(logvars))
    else:
      base_dist = tfp.distributions.Bernoulli(logits=params, dtype='float32')
    if self.any_periodic:
      this_dist = base_dist
    else:
      this_dist = tfp.distributions.Independent(base_dist, reinterpreted_batch_ndims=1)
    return this_dist

  def create_sample(self, param_mean,
                    param_logvar=None, sample_only=True, skip_input=None):
    """Convenient to have a function that, given a base parameter set, samples from
the autoregressive distribution by looping over the degress of freedom. Must use during
generation (no training data provided), but useful to have at other times as well.
    """
    if self.skips and skip_input is None:
      print("If using skip connections, need to provide latent tensor as skip_input.")
      print("Since skip_input=None, passing to self.autonet <should> throw an error.")
    if not self.skips and skip_input is not None:
      print("Skip connections are not in use, but skip_input is not None.")
      print("This will result in the input being ignored, so check this.")
    #If the autonet has conditional inputs, the first DOF group may also be modified
    #So first pass through autonet before breaking out first set of parameters
    #Technically, input won't actually matter, just conditional_input for this first pass
    #Because of handling of periodic DOFs, param_mean may be a list
    #Need to check this and create initial input based on correct shape
    if isinstance(param_mean, list):
      auto_in = tf.zeros_like(param_mean[0])
    else:
      auto_in = tf.zeros_like(param_mean)
    init_shift = self.autonet(auto_in, conditional_input=skip_input)
    if self.return_vars:
      mean_out, logvar_out = self._split_shift(param_mean, param_logvar, init_shift)
      logvar_out = logvar_out[:, :self.auto_group_size]
    else:
      mean_out = param_mean + tf.squeeze(init_shift, axis=-1)
    mean_out = mean_out[:, :self.auto_group_size]
    #Will need to pad to generate starting sample
    padding = [[0, 0], [0, np.prod(self.out_shape)-self.auto_group_size]]
    #Below will fail if return_vars is True but param_logvar is not specified!
    if self.return_vars:
      sample_out = self.create_dist([tf.pad(mean_out, padding),
                                     tf.pad(logvar_out, padding)]).sample()
      #If periodic DOFs, sample needs to be stacked together
      if self.any_periodic:
        sample_out = tf.stack(sample_out, axis=1)
    else:
      sample_out = self.create_dist(tf.pad(mean_out, padding)).sample()
    #Do in a loop over number of dimensions in data, sampling each based on previous
    #Stride it by the autoregressive group size, though
    for i in range(self.auto_group_size, np.prod(self.out_shape), self.auto_group_size):
      shift = self.autonet(sample_out, conditional_input=skip_input)
      if self.return_vars:
        this_mean, this_logvar = self._split_shift(param_mean, param_logvar, shift)
        logvar_out = tf.concat((logvar_out,
                                tf.gather(this_logvar,
                                          np.arange(i, i+self.auto_group_size),
                                          axis=-1)),
                                axis=-1)
      else:
        this_mean = param_mean + tf.squeeze(shift, axis=-1)
      #Add on to existing parameter information
      mean_out = tf.concat((mean_out,
                            tf.gather(this_mean,
                                      np.arange(i, i+self.auto_group_size),
                                      axis=-1)),
                            axis=-1)
      #Sample from distribution because need values to predict next degree of freedom
      if self.return_vars:
        #If want truncation, get it before sampling
        if self.truncate_normal:
          this_lowcut, this_highcut = self.get_truncation(this_mean, sample_out)
          this_sample = self.create_dist([this_mean, this_logvar,
                                          this_lowcut, this_highcut]).sample()
        else:
          this_sample = self.create_dist([this_mean, this_logvar]).sample()
          #Again stack back together if had any periodic DOFs
          if self.any_periodic:
            this_sample = tf.stack(this_sample, axis=1)
      else:
        this_sample = self.create_dist(this_mean).sample()
      sample_out = tf.concat((sample_out[:, :i], this_sample[:, i:]), axis=-1)
    if self.return_vars:
      if sample_only:
        return sample_out
      else:
        return mean_out, logvar_out, sample_out
    else:
      if sample_only:
        return sample_out
      else:
        return mean_out, sample_out

  def call(self, latent_tensor, train_data=None):
    #First just convert from latent to full-dimensional space
    d1_out = self.d1(latent_tensor)
    d2_out = self.d2(d1_out)
    param_mean = self.base_param(d2_out)
    if self.return_vars:
      if self.any_periodic:
        param_cos, param_sin, param_logvar = tf.split(param_mean, 3, axis=-1)
        param_mean = [param_cos, param_sin]
      else:
        param_mean, param_logvar = tf.split(param_mean, 2, axis=-1)
    #Next need to pass through autoregressive network
    #Do this differently if training or generating configurations
    #If training, training data should be provided, which will be passed through autonet
    #The result will be the parameters to be used to evaluate log probabilities of the data
    if train_data is not None:
      shift = self.autonet(self.flatten(train_data), conditional_input=latent_tensor)
      if self.return_vars:
        mean_out, logvar_out = self._split_shift(param_mean, param_logvar, shift)
        if self.truncate_normal:
          low_out, high_out = self.get_truncation(mean_out, self.flatten(train_data))
          mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
          logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
          low_out = tf.reshape(low_out, shape=(-1,)+self.out_shape)
          high_out = tf.reshape(high_out, shape=(-1,)+self.out_shape)
          return mean_out, logvar_out, low_out, high_out
        else:
          mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
          logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
          return mean_out, logvar_out
      else:
        mean_out = tf.reshape(param_mean + tf.squeeze(shift, axis=-1),
                              shape=(-1,)+self.out_shape)
        return mean_out
    #If not training draw sample based on output of dense layers
    #Much more expensive because loop over dimensions
    else:
      #If skip connections active, need to pass latent tensor to create_sample
      if self.skips:
        skip_input = latent_tensor
      else:
        skip_input = None
      if self.return_vars:
        mean_out, logvar_out, sample_out = self.create_sample(param_mean, param_logvar,
                                                    sample_only=False, skip_input=skip_input)
        if self.truncate_normal:
          low_out, high_out = self.get_truncation(mean_out, self.flatten(sample_out))
          mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
          logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
          low_out = tf.reshape(low_out, shape=(-1,)+self.out_shape)
          high_out = tf.reshape(high_out, shape=(-1,)+self.out_shape)
          sample_out = tf.reshape(sample_out, shape=(-1,)+self.out_shape)
          return mean_out, logvar_out, low_out, high_out, sample_out
        else:
          mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
          logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
          sample_out = tf.reshape(sample_out, shape=(-1,)+self.out_shape)
          return mean_out, logvar_out, sample_out
      else:
        mean_out, sample_out = self.create_sample(param_mean,
                                                  sample_only=False, skip_input=skip_input)
        mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
        sample_out = tf.reshape(sample_out, shape=(-1,)+self.out_shape)
        return mean_out, sample_out


class AutoConvDecoder(tf.keras.layers.Layer):
  """Convolutional autoregressive decoder. Wraps PixelCNN distribution in tfp.

  Args:
    latent_tensor: Input tensor to connect decoder to.
    out_shape: Shape of the data.
  """

  def __init__(self, out_shape, latent_dim, name='decoder',
               kernel_initializer='glorot_uniform',
               **kwargs):
    super(AutoConvDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.latent_dim = latent_dim
    self.kernel_initializer=kernel_initializer
    #Define PixelCNN parameters
    pixelcnn_params = {'dropout_p':0.3,
                       'num_resnet':1,
                       'num_hierarchies':2,
                       'num_filters':32,
                       'num_logistic_mix':2,
                       'receptive_field_dims':(4,4)}
    #Create convolutional autoregressive neural network
    self.autoconvnet = tfp.distributions.pixel_cnn._PixelCNNNetwork(**pixelcnn_params)
    convnet_input_shape = [(None,)+self.out_shape, (None, self.latent_dim)]
    self.autoconvnet.build(convnet_input_shape)
    self.pixelcnn = tfp.distributions.PixelCNN(self.out_shape,
                                               conditional_shape=(self.latent_dim,),
                                               high=1, low=0, **pixelcnn_params)
    #Replace the PixelCNN object's network with ours so we can access trainable weights
    self.pixelcnn.network = self.autoconvnet

  def call(self, latent_tensor, train_data=None):
    #If training data is passed, just go through the PixelCNN to get logP
    #Return that and use a special loss function
    if train_data is not None:
      prob_out = self.pixelcnn.log_prob(train_data,
                                        conditional_input=latent_tensor,
                                        training=True)
      return prob_out
    #If not training draw sample based solely on latent tensor
    #To make work with other autoregressive model outputs, need 2 returns
    #First return should be probability of each DOF taking a value of 1
    #But can't get at that info easily due to design of the PixelCNN, so just return 1's
    else:
      conf_out = self.pixelcnn.sample(latent_tensor.shape[0],
                                      conditional_input=latent_tensor,
                                      training=False)
      return tf.ones(conf_out.shape), conf_out


class AutoregressivePrior(tf.keras.layers.Layer):
  """Creates autoregressive model for the prior distribution, P(z), representing a direct
modelling alternative to a flow that can be used to represent discrete distributions. A full
neural network representation of the potential function for an Srel model would be more
general and expressive, but more difficult to code and train.
  """

  def __init__(self, out_shape, name='prior',
               hidden_dim=200,
               kernel_initializer='glorot_uniform',
               return_vars=False,
               **kwargs):
    super(AutoregressivePrior, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.return_vars = return_vars
    if self.return_vars:
      self.num_params = 2
    else:
      self.num_params = 1
    #Create variables to represent the parameters for the first DOF
    self.first_params = self.add_weight(shape=(self.num_params,), name='first_params',
                                        initializer=self.kernel_initializer,
                                        trainable=True)
    #Next create autoregressive model to predict probabilities
    #(Really parameters for probabilitistic models)
    self.auto_groupings = np.repeat(
                          np.arange(1, np.ceil(np.prod(out_shape)) + 1),
                          1)[:int(np.prod(out_shape))].astype(np.int32)
    self.autonet = MaskedNet(self.num_params,
                             event_shape=np.prod(self.out_shape),
                             hidden_units=[self.hidden_dim,],
                             dof_order=self.auto_groupings,
                             activation='tanh',
                             use_bias=True,
                             kernel_initializer=self.kernel_initializer)
    #And will need function to flatten training data if have it
    self.flatten = tf.keras.layers.Flatten()

  def create_dist(self, params):
    """Need a function to create a sampling distribution for the autoregressive distribution.
Will use Gaussian if return_vars is True and Bernoulli if False. Note that during training
no sampling is performed, only calculation of probabilities, allowing gradients to work.
Really just want to pass parameters - can use autoregressive network outside of this.
If return_vars is true, params should be a list of [means, logvars].
    """
    if self.return_vars:
      means = params[0]
      logvars = params[1]
      base_dist = tfp.distributions.Normal(means, tf.exp(0.5*logvars))
      #base_dist = tfp.distributions.Laplace(means, tf.exp(logvars))
    else:
      base_dist = tfp.distributions.Bernoulli(logits=params, dtype='float32')
    this_dist = tfp.distributions.Independent(base_dist, reinterpreted_batch_ndims=1)
    return this_dist

  def call(self, batch_size, train_data=None):
    """Only input is batch size (how many samples to produce).
If training data provided, overrides.
    """
    if train_data is not None:
      train_data = self.flatten(train_data)
      batch_size = train_data.shape[0]

    #Assign parameters for first DOF across all batch members
    first_params = tf.tile(tf.reshape(self.first_params, (1,)+self.first_params.shape),
                           (batch_size, 1))

    #If provided training data, use to find parameters from autoregressive model
    if train_data is not None:
      means = self.autonet(self.flatten(train_data))
      if self.return_vars:
        means, logvars = tf.split(means, 2, axis=-1)
        logvars = tf.squeeze(logvars, axis=-1)
        logvars = tf.concat((first_params[:, -1:], logvars[:, 1:]), axis=-1)
      #Have to replace meaningless first parameter
      means = tf.squeeze(means, axis=-1)
      means = tf.concat((first_params[:, :1], means[:, 1:]), axis=-1)
      #If just training, will return only calculated log probabilities
      if self.return_vars:
        log_probs = self.create_dist([means, logvars]).log_prob(train_data)
      else:
        log_probs = self.create_dist(means).log_prob(train_data)
      return log_probs

    #If not training, generate a sample and associated log probabilities
    else:
      mean_out = first_params[:, :1]
      if self.return_vars:
        logvar_out = first_params[:, -1:]
      #Will need to pad to generate starting sample
      padding = [[0, 0], [0, np.prod(self.out_shape)-1]]
      #Below will fail if return_vars is True but param_logvar is not specified!
      if self.return_vars:
        sample_out = self.create_dist([tf.pad(mean_out, padding),
                                       tf.pad(logvar_out, padding)]).sample()
      else:
        sample_out = self.create_dist(tf.pad(mean_out, padding)).sample()
      #Do in a loop over number of dimensions in data, sampling each based on previous
      for i in range(1, np.prod(self.out_shape)):
        this_means = self.autonet(sample_out)
        if self.return_vars:
          this_means, this_logvars = tf.split(this_means, 2, axis=-1)
          this_logvars = tf.squeeze(this_logvars, axis=-1)
          logvar_out = tf.concat((logvar_out, this_logvars[:, i:i+1]), axis=-1)
        #Add on to existing parameter information
        this_means = tf.squeeze(this_means, axis=-1)
        mean_out = tf.concat((mean_out, this_means[:, i:i+1]), axis=-1)
        #Sample from distribution because need values to predict next degree of freedom
        if self.return_vars:
          this_sample = self.create_dist([this_means, this_logvars]).sample()
        else:
          this_sample = self.create_dist(this_means).sample()
        sample_out = tf.concat((sample_out[:, :i], this_sample[:, i:]), axis=-1)
      #Calculate log probabilities and return
      if self.return_vars:
        log_probs = self.create_dist([mean_out, logvar_out]).log_prob(sample_out)
      else:
        log_probs = self.create_dist(mean_out).log_prob(sample_out)
      return log_probs, tf.reshape(sample_out, (batch_size,)+self.out_shape)


class MaskedSplineBijector(tf.keras.layers.Layer):
  """Follows tfp example for using rational quadratic splines (as described in Durkan et al.
2019) to replace affine transformations (in, say, RealNVP). This should allow more flexible
transformations with similar cost and should work much better with 1D flows. This version
uses dense layers that are masked to be autoregressive with conditional inputs optional.
  """

  def _bin_positions(self, x):
    #x = tf.reshape(x, [x.shape[0], -1, self.num_bins])
    #MaskedNet already reshapes to event_shape by num_params
    #So providing event_shape=data_dim and num_params=num_bins
    out = tf.math.softmax(x, axis=-1)
    out = out*(self.bin_max - self.bin_min - self.num_bins*1e-2) + 1e-2
    return out

  def _slopes(self, x):
    #x = tf.reshape(x, [x.shape[0], -1, self.num_bins - 1])
    return tf.math.softplus(x) + 1e-2

  def __init__(self, data_dim, name='rqs',
               bin_range=[-10.0, 10.0],
               num_bins=32, hidden_dim=200,
               kernel_initializer='truncated_normal',
               conditional=False,
               conditional_event_shape=None,
               dof_order=None,
               **kwargs):
    super(MaskedSplineBijector, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.bin_min = bin_range[0]
    self.bin_max = bin_range[1]
    self.num_bins = num_bins
    self.hidden_dim = hidden_dim
    self.kernel_initializer = kernel_initializer
    self.conditional = conditional
    self.conditional_event_shape = conditional_event_shape   
    self.dof_order = dof_order
    #Don't create an initial masked neural net layer...
    #Will add hidden layers for each spline parameter-generating network
    #Since autoregressive, 1st DOF will only depend on conditional input
    #If apply two MaskedNet in a row, end up with the second DOF also only depending on conditional inputs
    #And so on if have more
    #(because the 2nd DOF output from the second net only depends on the 1st DOF output from the first net)
    #Makes sense if remember that MaskedNet is only designed to generate outputs same dimension as inputs
    #(multiplied by num_params, which is the first argument and set to 1)
    #So create neural nets for widths, heights, and slopes
    #And to get right output dimensionality, make self.num_bins the number of parameters
    self.bin_widths = MaskedNet(self.num_bins,
                                  event_shape=self.data_dim,
                                  conditional=self.conditional,
                                  conditional_event_shape=self.conditional_event_shape,
                                  hidden_units=[self.hidden_dim,],
                                  dof_order=self.dof_order,
                                  activation=tf.nn.tanh,
                                  use_bias=True,
                                  kernel_initializer=self.kernel_initializer,
                                  name='w',
                                 )
    self.bin_heights = MaskedNet(self.num_bins,
                                   event_shape=self.data_dim,
                                   conditional=self.conditional,
                                   conditional_event_shape=self.conditional_event_shape,
                                   hidden_units=[self.hidden_dim,],
                                   dof_order=self.dof_order,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=self.kernel_initializer,
                                   name='h',
                                  )
    self.knot_slopes = MaskedNet(self.num_bins - 1,
                                   event_shape=self.data_dim,
                                   conditional=self.conditional,
                                   conditional_event_shape=self.conditional_event_shape,
                                   hidden_units=[self.hidden_dim,],
                                   dof_order=self.dof_order,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=self.kernel_initializer,
                                   name='s',
                                  )

  def call(self, input_tensor, conditional_input=None):
    if input_tensor.shape[1] == 0:
      input_tensor = tf.ones((input_tensor.shape[0], 1))
    #Use nets to get spline parameters
    bw = self.bin_widths(input_tensor, conditional_input=conditional_input)
    #Apply "activations" manually since MaskedNet does not apply activation on last layer
    bw = self._bin_positions(bw)
    bh = self.bin_heights(input_tensor, conditional_input=conditional_input)
    bh = self._bin_positions(bh)
    ks = self.knot_slopes(input_tensor, conditional_input=conditional_input)
    ks = self._slopes(ks)
    return tfp.bijectors.RationalQuadraticSpline(bin_widths=bw,
                                                 bin_heights=bh,
                                                 knot_slopes=ks,
                                                 range_min=self.bin_min)


class NormFlowRQSplineMAF(tf.keras.layers.Layer):
  """Follows tfp example for using rational quadratic splines (as described in Durkan et al.
2019) but with masked autoregressive flow (MAF) structure.
  """

  def __init__(self, data_dim, name='rqs_maf', num_blocks=2,
               kernel_initializer='truncated_normal', rqs_params={},
               **kwargs):
    super(NormFlowRQSplineMAF, self).__init__(name=name, **kwargs)
    self.data_dim = data_dim
    self.num_blocks = num_blocks
    self.kernel_initializer = kernel_initializer
    self.rqs_params = rqs_params
    #Want to create a spline bijector for each block, so create lists and loop
    self.net_list = []
    self.block_list = []
    for l in range(self.num_blocks):
      self.net_list.append(MaskedSplineBijector(data_dim,
                                          kernel_initializer=self.kernel_initializer,
                                          **rqs_params))
      self.block_list.append(tfp.bijectors.MaskedAutoregressiveFlow(
                                                   bijector_fn=self.net_list[l],
                                                   name='block_%i'%l))

  def call(self, input_tensor, reverse=False, **kwargs):
    out = input_tensor
    log_det_sum = tf.zeros(tf.shape(input_tensor)[0])
    if not reverse:
      for block in self.block_list:
        log_det_sum += block.forward_log_det_jacobian(out, **kwargs,
                                                      event_ndims=len(input_tensor.shape)-1)
        out = block.forward(out, **kwargs)
    else:
      for block in self.block_list[::-1]:
        log_det_sum += block.inverse_log_det_jacobian(out, **kwargs,
                                                      event_ndims=len(input_tensor.shape)-1)
        out = block.inverse(out, **kwargs)
    return out, log_det_sum


class AutoregressiveFlowDecoder(tf.keras.layers.Layer):
  """Autoregressive decoder with flow after sampling to allow most flexible probability
modeling when decoding. Essentially works like normal decoding, mapping latents to
parameters for Gaussian or von Mises distributions. But just does this without any
autoregression. To build in autoregression, and to make probability distribution very
flexible, flow sample through autoregressive flow with masked rational quadratic spline
bijectors. Should allow for multimodal distributions with autoregression. Note that it
will be conditioned on the latent input throughout, so really its a conditional flow.
  """

  def __init__(self, out_shape, name='decoder',
               hidden_dim=200,
               kernel_initializer='glorot_uniform',
               auto_group_size=1,
               periodic_dofs=[False,],
               non_periodic_data_range=20.0,
               **kwargs):
    super(AutoregressiveFlowDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.hidden_dim = hidden_dim
    self.kernel_initializer=kernel_initializer
    #Set return_vars so will be consistent with other decoders
    self.return_vars = True
    #Specify which DOFs should be periodic and use VonMises distribution
    self.any_periodic = np.any(periodic_dofs)
    if self.any_periodic:
      #Check if number of specified DOFs match output shape
      if np.prod(out_shape) != len(periodic_dofs):
        raise ValueError('Length of periodic_dofs does not match flattened out_shape.'
                         +'\nIf specifying any periodic DOFs, must specify for all DOFs')
      else:
        self.periodic_dofs = periodic_dofs
    else:
      self.periodic_dofs = [False,]*np.prod(out_shape)
    #Specify how to group DOFs together when autoregressing based on group size
    self.auto_group_size = auto_group_size
    #For all separate (default if None when pass to MaskedNet) get [1, 2, 3, 4,...]
    #If wanted to group two DOFs at a time, would want [1, 1, 2, 2, 3, 3,...]
    self.auto_groupings = np.repeat(
                          np.arange(1, np.ceil(np.prod(out_shape)/self.auto_group_size) + 1),
                          self.auto_group_size)[:int(np.prod(out_shape))].astype(np.int32)
    #Create initial dense neural network layer
    self.d1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.tanh,
                                    kernel_initializer=self.kernel_initializer)
    #Assume sampling distributions before flow are Gaussian or von Mises
    #So will only have 2 parameters
    self.out_event_dims = 2
    if self.any_periodic:
      self.out_event_dims += 1
    #Will predict means (and log variances) with neural net
    #Autoregressive flow will then map what gets sampled
    self.base_param = tf.keras.layers.Dense(self.out_event_dims*np.prod(self.out_shape),
                                            activation=None,
                                            kernel_initializer=self.kernel_initializer)
    #And will need function to flatten training data if have it
    self.flatten = tf.keras.layers.Flatten()
    #To handle periodic DOFs and allow to pass through flow and maintain domain,
    #need to scale all DOFs so can just use -pi to pi for flow domain for all
    #So define scaling here based on supplied non_periodic_data_range
    scale_for_flow = np.ones(len(periodic_dofs), dtype='float32')
    scale_for_flow[~np.array(periodic_dofs)] = 2.0*np.pi/non_periodic_data_range
    self.scale_for_flow = tf.convert_to_tensor(scale_for_flow)

  def build(self, input_shape):
    #To set up masked autoregressive flow, need to know output dimension
    #And latent dimension for conditional inputs
    #Masked autoregressive flow with spline bijectors
    self.maf = NormFlowRQSplineMAF(np.prod(self.out_shape),
                                   num_blocks=2,
                                   rqs_params={'hidden_dim':self.hidden_dim,
                                               'conditional':True,
                                               'conditional_event_shape':input_shape[1:],
                                               'dof_order':self.auto_groupings,
                                               'bin_range':[-np.pi, np.pi]})
    #Note that MUST make bin range -pi to pi to accomodate periodic DOFs
    #For other DOFs, will just scale from non_periodic_data_range to -pi to pi
    super(AutoregressiveFlowDecoder, self).build(input_shape)

  def _transform_params(self, params):
    """Two ways to define base parameters depending on periodic DOFs. If periodic, want
to define mean of von Mises as angle defined by sine and cosine outputs, which keeps it
in the proper domain.
    """
    params = tf.reshape(params, (-1, tf.reduce_prod(self.out_shape), self.out_event_dims))
    #out_event_dims should be 3 for periodic and 2 for non-periodic
    if self.any_periodic:
      cos_mean, sin_mean, logvar = tf.squeeze(tf.split(params, 3, axis=-1), axis=-1)
      #For periodic DOFs, pass through arctan2
      mean_out_p = tf.math.atan2(sin_mean, cos_mean)
      #For non-periodic, just add together
      mean_out_nonp = sin_mean + cos_mean
      #Take only the elements we want from each (wastes comp, but differentiable and works)
      mean_out = tf.where(self.periodic_dofs, mean_out_p, mean_out_nonp)
      logvar_out = logvar
    else:
      mean_out, logvar_out = tf.squeeze(tf.split(params, 2, axis=-1), axis=-1)
    return mean_out, logvar_out

  def create_dist(self, params):
    """Need a function to create a sampling distribution, using Gaussian for anything
not periodic and von Mises for anything periodic. Note that during training, no sampling
is performed, only calculation of probabilities, allowing gradients to work.
    """
    means = params[0]
    logvars = params[1]
    #For periodic DOFs, need to build list of distributions
    #For 'True' values, use VonMises, for 'False' use Normal
    if self.any_periodic:
      base_dist_list = []
      for i in range(means.shape[1]):
        if self.periodic_dofs[i]:
          #With VonMises in tfp, will get NaN if logvars too negative
          #Issues with sampling if less than about -43 and issues with logP if about -80
          #Doesn't give infinity, though, just NaN due to sampling algorithm
          #So we will make sure it's bigger than -40
          #Well, those numbers work on a CPU... for a GPU you get random NaNs
          #starting at about -17 for the purposes of sampling
          base_dist_list.append(tfp.distributions.VonMises(means[:, i],
                                              tf.exp(-tf.maximum(logvars[:, i], -15.0))))
        else:
          base_dist_list.append(tfp.distributions.Normal(means[:, i],
                                                         tf.exp(0.5*logvars[:, i])))
      #Join together into a joint distribution
      #This is not as nice to work with as Independent, but will be ok
      #log_prob() function will return in same way
      #But sample() will give list of tensors for each DOF, so will need to stack them
      #(and unstack to feed into log_prob())
      base_dist = tfp.distributions.JointDistributionSequential(base_dist_list)
      this_dist = base_dist
    else:
      base_dist = tfp.distributions.Normal(means, tf.exp(0.5*logvars))
      this_dist = tfp.distributions.Independent(base_dist, reinterpreted_batch_ndims=1)
    return this_dist

  def call(self, latent_tensor, train_data=None):
    #First predict base parameters with standard dense networks
    d1_out = self.d1(latent_tensor)
    params = self.base_param(d1_out)
    #Transform parameters as needed for periodic DOFs
    mean_out, logvar_out = self._transform_params(params)
    #If we are training (train_data not None), need to evaluate log probability and return
    #This is because inverse flow requires knowledge of the latent coordinate
    #So makes more sense to just also provide log probability as extra information
    if train_data is not None:
      rev_train_data, rev_logdet = self.maf(train_data*self.scale_for_flow,
                                            conditional_input=latent_tensor,
                                            reverse=True)
      rev_train_data = rev_train_data/self.scale_for_flow
      #Need to unstack data if have periodic DOFs
      if self.any_periodic:
        rev_train_data = tf.unstack(rev_train_data, axis=1)
      log_p = self.create_dist([mean_out, logvar_out]).log_prob(rev_train_data)
      log_p = log_p + rev_logdet
      mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
      logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
      return mean_out, logvar_out, log_p
    #If generating (so train_data=None), actually sample parameters, then flow
    else:
      #First generate distributions from parameters
      prob_dist = self.create_dist([mean_out, logvar_out])
      sample_out = prob_dist.sample()
      log_p = prob_dist.log_prob(sample_out)
      #Need to stack data if have periodic DOFs
      if self.any_periodic:
        sample_out = tf.stack(sample_out, axis=1)
      sample_out = sample_out*self.scale_for_flow
      #Pass through forward flow
      sample_out, logdet_out = self.maf(sample_out,
                                        conditional_input=latent_tensor,
                                        reverse=False)
      log_p = log_p - logdet_out
      sample_out = sample_out/self.scale_for_flow
      mean_out = tf.reshape(mean_out, shape=(-1,)+self.out_shape)
      logvar_out = tf.reshape(logvar_out, shape=(-1,)+self.out_shape)
      sample_out = tf.reshape(sample_out, shape=(-1,)+self.out_shape)
      #To work with MC moves, just return log_p and create AutoregressiveLoss function
      #Works similarly to other autoregressive models because only thing added
      #when not training is the actual sample, which is always returned last
      return mean_out, logvar_out, log_p, sample_out


