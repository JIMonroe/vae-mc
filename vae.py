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

"""
Implementation of VAE based models for unsupervised learning of disentangled
representations.
Adapted from disentanglement_lib https://github.com/google-research/disentanglement_lib
"""

import math
from libVAE import architectures, losses
import tensorflow as tf


class BaseVAE(tf.keras.Model):
  """Abstract base class of a basic VAE."""

  def __init__(self, data_shape, num_latent,
               name='vae', arch='fc', include_vars=False, **kwargs):
    super(BaseVAE, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.include_vars = include_vars
    #By default, use fully-connect (fc) architecture for neural nets
    #Can switch to convolutional if specify arch='conv'
    self.arch = arch
    if self.arch == 'conv':
      self.encoder = architectures.ConvEncoder(num_latent)
      self.decoder = architectures.DeconvDecoder(data_shape)
    else:
      self.encoder = architectures.FCEncoder(num_latent)
      self.decoder = architectures.FCDecoder(data_shape, return_vars=self.include_vars)
    self.sampler = architectures.SampleLatent()

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    #For basic VAE, just return kl_loss (i.e. beta=1)
    return 1.0*kl_loss

  def call(self, inputs):
    z_mean, z_logvar = self.encoder(inputs)
    z = self.sampler(z_mean, z_logvar)
    reconstructed = self.decoder(z)
    #Note that if include_vars is True reconstructed will be a tuple of (means, log_vars)
    kl_loss = losses.compute_gaussian_kl(z_mean, z_logvar)
    reg_loss = self.regularizer(kl_loss, z_mean, z_logvar, z)
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class BetaVAE(BaseVAE):
  """BetaVAE model."""

  def __init__(self, data_shape, num_latent, beta=6.0, name='beta_vae', **kwargs):
    """Creates a beta-VAE model.

    Implementing Eq. 4 of "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl).

    Args:
      beta: Hyperparameter for the regularizer.

    Returns:
      model_fn: Model function for TPUEstimator.
    """
    super(BetaVAE, self).__init__(data_shape, num_latent, name=name, **kwargs)
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    return self.beta * kl_loss


def compute_covariance_z_mean(z_mean):
  """Computes the covariance of z_mean.

  Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.

  Args:
    z_mean: Encoder mean, tensor of size [batch_size, num_latent].

  Returns:
    cov_z_mean: Covariance of encoder mean, tensor of size [num_latent,
      num_latent].
  """
  expectation_z_mean_z_mean_t = tf.reduce_mean(
      tf.expand_dims(z_mean, 2) * tf.expand_dims(z_mean, 1), axis=0)
  expectation_z_mean = tf.reduce_mean(z_mean, axis=0)
  cov_z_mean = tf.subtract(
      expectation_z_mean_z_mean_t,
      tf.expand_dims(expectation_z_mean, 1) * tf.expand_dims(
          expectation_z_mean, 0))
  return cov_z_mean


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
  """Compute on and off diagonal regularizers for DIP-VAE models.

  Penalize deviations of covariance_matrix from the identity matrix. Uses
  different weights for the deviations of the diagonal and off diagonal entries.

  Args:
    covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
    lambda_od: Weight of penalty for off diagonal elements.
    lambda_d: Weight of penalty for diagonal elements.

  Returns:
    dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
  """
  covariance_matrix_diagonal = tf.linalg.diag_part(covariance_matrix)
  covariance_matrix_off_diagonal = covariance_matrix - tf.linalg.diag(
      covariance_matrix_diagonal)
  dip_regularizer = tf.add(
      lambda_od * tf.reduce_sum(covariance_matrix_off_diagonal**2),
      lambda_d * tf.reduce_sum((covariance_matrix_diagonal - 1)**2))
  return dip_regularizer


class DIPVAE(BaseVAE):
  """DIPVAE model."""

  def __init__(self, data_shape, num_latent, lambda_od=20.0, lambda_d_factor=1.0, dip_type="ii", name='dip_vae', **kwargs):
    """Creates a DIP-VAE model.

    Based on Equation 6 and 7 of "Variational Inference of Disentangled Latent
    Concepts from Unlabeled Observations"
    (https://openreview.net/pdf?id=H1kG7GZAW).

    Args:
      lambda_od: Hyperparameter for off diagonal values of covariance matrix.
      lambda_d_factor: Hyperparameter for diagonal values of covariance matrix
                       (set to 10.0 for type 'i' and 1.0 for type 'ii')
        lambda_d = lambda_d_factor*lambda_od.
      dip_type: "i" or "ii".
    """
    super(DIPVAE, self).__init__(data_shape, num_latent, name=name, **kwargs)
    self.lambda_od = lambda_od
    self.lambda_d_factor = lambda_d_factor
    self.dip_type = dip_type

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    cov_z_mean = compute_covariance_z_mean(z_mean)
    lambda_d = self.lambda_d_factor * self.lambda_od
    if self.dip_type == "i":  # Eq 6 page 4
      # mu = z_mean is [batch_size, num_latent]
      # Compute cov_p(x) [mu(x)] = E[mu*mu^T] - E[mu]E[mu]^T]
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z_mean, self.lambda_od, lambda_d)
    elif self.dip_type == "ii":
      cov_enc = tf.linalg.diag(tf.exp(z_logvar))
      expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
      cov_z = expectation_cov_enc + cov_z_mean
      cov_dip_regularizer = regularize_diag_off_diag_dip(
          cov_z, self.lambda_od, lambda_d)
    else:
      raise NotImplementedError("DIP variant not supported.")
    return kl_loss + cov_dip_regularizer


def gaussian_log_density(samples, mean, log_var):
  pi = tf.constant(math.pi)
  normalization = tf.math.log(2. * pi)
  inv_sigma = tf.exp(-log_var)
  tmp = (samples - mean)
  return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
  """Estimate of total correlation on a batch.

  We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
  log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
  for the minimization. The constant should be equal to (num_latents - 1) *
  log(batch_size * dataset_size)

  Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.

  Returns:
    Total correlation estimated on a batch.
  """
  # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
  # tensor of size [batch_size, batch_size, num_latents]. In the following
  # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
  log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
      tf.expand_dims(z_logvar, 0))
  # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
  # + constant) for each sample in the batch, which is a vector of size
  # [batch_size,].
  log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
  # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
  # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
  log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
  return tf.reduce_mean(log_qz - log_qz_product)


class BetaTCVAE(BaseVAE):
  """BetaTCVAE model."""

  def __init__(self, data_shape, num_latent, beta=6.0, name='beta_tc_vae', **kwargs):
    """Creates a beta-TC-VAE model.

    Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of
    Disentanglement in Variational Autoencoders"
    (https://arxiv.org/pdf/1802.04942).
    If alpha = gamma = 1, Eq. 4 can be written as ELBO + (1 - beta) * TC.

    Args:
      beta: Hyperparameter total correlation.
    """
    super(BetaTCVAE, self).__init__(data_shape, num_latent, name=name, **kwargs)
    self.beta = beta

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    tc = (self.beta - 1.) * total_correlation(z_sampled, z_mean, z_logvar)
    return tc + kl_loss


class FlowVAE(tf.keras.Model):
  """VAE with normalizing flow on the latent space."""

  def __init__(self, data_shape, num_latent,
               name='flow_vae', arch='fc', include_vars=False,
               **kwargs):
    super(FlowVAE, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.include_vars = include_vars
    #By default, use fully-connect (fc) architecture for neural nets
    #Can switch to convolutional if specify arch='conv' (won't have flow, though)
    self.arch = arch
    flow_net_params = {'num_hidden':2, 'hidden_dim':200}
    if self.arch == 'conv':
      self.encoder = architectures.ConvEncoder(num_latent)
      self.decoder = architectures.DeconvDecoder(data_shape)
    else:
      self.encoder = architectures.FCEncoderFlow(num_latent, hidden_dim=1200,
                                                 flow_net_params=flow_net_params)
      self.decoder = architectures.FCDecoder(data_shape, return_vars=self.include_vars)
    self.sampler = architectures.SampleLatent()
    self.flow = architectures.NormFlow(num_latent, flow_net_params=flow_net_params)

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    #For basic VAE, just return kl_loss (i.e. beta=1)
    return 1.0*kl_loss

  def call(self, inputs):
    z_mean, z_logvar, uv, b = self.encoder(inputs)
    z = self.sampler(z_mean, z_logvar)
    tz, logdet = self.flow(z, uv_list=uv, b_list=b)
    reconstructed = self.decoder(tz)
    #Note that if include_vars is True reconstructed will be a tuple of (means, log_vars)
    #Estimate the KL divergence - should return average KL over batch
    kl_loss = losses.estimate_gaussian_kl(tz, z, z_mean, z_logvar)
    #And subtract the average log determinant for the flow transformation
    kl_loss -= tf.reduce_mean(logdet)
    reg_loss = self.regularizer(kl_loss, z_mean, z_logvar, z)
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


