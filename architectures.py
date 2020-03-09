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

  def __init__(self, num_latent, name='encoder', **kwargs):
    super(FCEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.flattened = tf.keras.layers.Flatten()
    self.e1 = tf.keras.layers.Dense(1200, activation=tf.nn.relu, name="e1")
    self.e2 = tf.keras.layers.Dense(1200, activation=tf.nn.relu, name="e2")
    self.means = tf.keras.layers.Dense(num_latent, activation=None)
    self.log_var = tf.keras.layers.Dense(num_latent, activation=None)

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

  def __init__(self, num_latent, name='encoder', **kwargs):
    super(ConvEncoder, self).__init__(name=name, **kwargs)
    self.num_latent = num_latent
    self.e1 = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=4,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e1",
                                    )
    self.e2 = tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=4,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e2",
                                    )
    self.e3 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=2,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e3",
                                    )
    self.e4 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=2,
                                     strides=2,
                                     activation=tf.nn.relu,
                                     padding="same",
                                     name="e4",
                                    )
    self.flat_e4 = tf.keras.layers.Flatten()
    self.e5 = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="e5")
    self.means = tf.keras.layers.Dense(num_latent, activation=None, name="means")
    self.log_var = tf.keras.layers.Dense(num_latent, activation=None, name="log_var")

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

  def __init__(self, out_shape, name='decoder', **kwargs):
    super(FCDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.d1 = tf.keras.layers.Dense(1200, activation=tf.nn.tanh)
    self.d2 = tf.keras.layers.Dense(1200, activation=tf.nn.tanh)
    self.d3 = tf.keras.layers.Dense(1200, activation=tf.nn.tanh)
    self.d4 = tf.keras.layers.Dense(np.prod(out_shape))

  def call(self, latent_tensor):
    d1_out = self.d1(latent_tensor)
    d2_out = self.d2(d1_out)
    d3_out = self.d3(d2_out)
    d4_out = self.d4(d3_out)
    return tf.reshape(d4_out, shape=(-1,) + self.out_shape)


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

  def __init__(self, out_shape, name='decoder', **kwargs):
    super(DeconvDecoder, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.d1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    self.d2 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
    self.d3 = tf.keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                             )
    self.d4 = tf.keras.layers.Conv2DTranspose(filters=32,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                             )
    self.d5 = tf.keras.layers.Conv2DTranspose(filters=32,
                                              kernel_size=4,
                                              strides=2,
                                              activation=tf.nn.relu,
                                              padding="same",
                                             )
    self.d6 = tf.keras.layers.Conv2DTranspose(filters=out_shape[2],
                                              kernel_size=4,
                                              strides=2,
                                              padding="same",
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


