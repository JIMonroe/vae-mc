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
import numpy as np
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
               beta=1.0,
               **kwargs):
    super(FlowVAE, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.include_vars = include_vars
    self.beta = beta
    #By default, use fully-connect (fc) architecture for neural nets
    #Can switch to convolutional if specify arch='conv' (won't have flow, though)
    self.arch = arch
    flow_net_params = {'num_hidden':2, 'hidden_dim':200,
                       'nvp_split':True, 'activation':tf.nn.relu}
    if self.arch == 'conv':
      self.encoder = architectures.ConvEncoder(num_latent)
      self.decoder = architectures.DeconvDecoder(data_shape)
    else:
      #self.encoder = architectures.FCEncoderFlow(num_latent, hidden_dim=1200,
      #                                           kernel_initializer='zeros',
      #                                           flow_net_params=flow_net_params)
      #Issue with predicting parameters with encoder and passing along...
      #Somehow these parameters aren't tracked when go through the ODE solver
      #The explicitly added weights (w and b) in the kernel network do get tracked
      #Must be something to do with variable scope or querying trainable parameters
      self.encoder = architectures.FCEncoder(num_latent, hidden_dim=1200)
      self.decoder = architectures.FCDecoder(data_shape, return_vars=self.include_vars)
    self.sampler = architectures.SampleLatent()
    self.flow = architectures.NormFlowRealNVP(num_latent,
                                              kernel_initializer='truncated_normal',
                                              flow_net_params=flow_net_params)

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    #For basic VAE, beta = 1.0, but want ability to change it
    return self.beta * kl_loss

  def call(self, inputs):
    z_mean, z_logvar = self.encoder(inputs)
    z = self.sampler(z_mean, z_logvar)
    tz, logdet = self.flow(z)
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


class PriorFlowVAE(tf.keras.Model):
  """VAE with normalizing flow on the prior, as suggested by Chen, et al. 2017 in
their paper 'Variational Lossy Autoencoder.'
  """

  def __init__(self, data_shape, num_latent,
               name='priorflow_vae', arch='fc',
               autoregress=False, include_vars=False,
               beta=1.0, flow_type='rqs',
               e_hidden_dim=1200,
               f_hidden_dim=200,
               d_hidden_dim=1200,
               use_skips=True,
               n_auto_group=1,
               truncate_norm=False,
               periodic_dof_inds=[],
               sample_latent=True,
               **kwargs):
    super(PriorFlowVAE, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.autoregress = autoregress
    self.include_vars = include_vars
    self.beta = beta
    self.e_hidden_dim = e_hidden_dim
    self.f_hidden_dim = f_hidden_dim
    self.d_hidden_dim = d_hidden_dim
    self.sample_latent = True
    #By default, use fully-connect (fc) architecture for neural nets
    #Can switch to convolutional if specify arch='conv' (won't have flow, though)
    self.arch = arch
    self.use_skips = use_skips
    self.n_auto_group = n_auto_group
    #Can truncate normal distributions, but only makes sense for particle dimer!
    self.truncate_norm = truncate_norm
    #Allow periodic DOFs using VonMises distribution instead of Normal
    #This specifies their indices and builds boolean mask to pass to AutoregressiveDecoder
    self.periodic_dofs = [False,]*data_shape[0]
    for i in periodic_dof_inds:
      self.periodic_dofs[i] = True
    if self.arch == 'conv':
      self.encoder = architectures.ConvEncoder(num_latent)
      if self.autoregress:
        #Not compatible with Gaussian P(x|z), so disable
        self.include_vars = False
        self.decoder = architectures.AutoConvDecoder(self.data_shape, self.num_latent)
      else:
        self.decoder = architectures.DeconvDecoder(self.data_shape)
    else:
      self.encoder = architectures.FCEncoder(num_latent, hidden_dim=self.e_hidden_dim,
                                             periodic_dofs=self.periodic_dofs)
      if self.autoregress:
        self.decoder = architectures.AutoregressiveDecoder(data_shape,
                                                           hidden_dim=self.d_hidden_dim,
                                                           return_vars=self.include_vars,
                                                           skip_connections=self.use_skips,
                                                           auto_group_size=self.n_auto_group,
                                                           truncate_normal=self.truncate_norm,
                                                           periodic_dofs=self.periodic_dofs)
      else:
        self.decoder = architectures.FCDecoder(data_shape, return_vars=self.include_vars,
                                               hidden_dim=self.d_hidden_dim)
    self.sampler = architectures.SampleLatent()
    if flow_type == 'affine':
      flow_net_params = {'num_hidden':2, 'hidden_dim':self.f_hidden_dim,
                         'nvp_split':True, 'activation':tf.nn.relu}
      self.flow = architectures.NormFlowRealNVP(num_latent,
                                                kernel_initializer='truncated_normal',
                                                flow_net_params=flow_net_params,
                                                num_blocks=4)
    #Default is rqs, but lazily don't catch if put something other than affine or rqs
    else:
      flow_net_params = {'bin_range':[-10.0, 10.0], 'num_bins':32,
                         'hidden_dim':self.f_hidden_dim}
      self.flow = architectures.NormFlowRQSplineRealNVP(self.num_latent,
                                                        kernel_initializer='truncated_normal',
                                                        rqs_params=flow_net_params)

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    #For basic VAE, beta = 1.0, but want ability to change it
    return self.beta * kl_loss

  def call(self, inputs, training=False):
    z_mean, z_logvar = self.encoder(inputs)
    #To allow regular autoencoder, can set sample_latent to False
    if self.sample_latent:
      z = self.sampler(z_mean, z_logvar)
    else:
      z = z_mean + z_logvar #Add together to keep same encoder size but use info differently
    #With flow only on prior, z passes directly through
    if self.autoregress and training:
      reconstructed = self.decoder(z, train_data=inputs)
    else:
      reconstructed = self.decoder(z)
    #Note that if include_vars is True reconstructed will be a tuple of (means, log_vars)
    #Before estimating KL divergence, pass z through inverse flow
    #(forward flow is used during generation from a standard normal)
    #May not actually need to do this DURING training... may be able to do after - will check
    #If do after, MUST do really well before actually using model in MC simulations
    #Feels weird, though, because if don't do, completely leave off KL term...
    if self.beta != 0.0:
      #This works with current style of VAE (beta-VAE), but if switch, may not
      z_prior, logdet = self.flow(z, reverse=True)
      #Estimate the KL divergence - should return average KL over batch
      kl_loss = losses.estimate_gaussian_kl(z_prior, z, z_mean, z_logvar)
      #And SUBTRACT the average log determinant for the flow transformation
      kl_loss -= tf.reduce_mean(logdet)
    else:
      kl_loss = 0.0
    reg_loss = self.regularizer(kl_loss, z_mean, z_logvar, z)
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class AdversarialVAE(tf.keras.Model):
  """VAE with adversarial discriminator instead of analytic KL loss."""

  def __init__(self, data_shape, num_latent,
               name='ad_vae', include_vars=False,
               beta=1.0,
               **kwargs):
    super(AdversarialVAE, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.include_vars = include_vars
    self.beta = beta
    self.encoder = architectures.AdversarialEncoder(num_latent,
                                                    hidden_dim_x=1200, hidden_dim_e=200)
    self.decoder = architectures.FCDecoder(data_shape, return_vars=self.include_vars)
    self.discriminator = architectures.DiscriminatorNet(hidden_dim_x=1200, hidden_dim_z=200)

  def regularizer(self, kl_loss):
    #For basic VAE, beta = 1.0, but want ability to change it
    return self.beta * kl_loss

  def call(self, inputs):
    #Encoder here generates random numbers, so do not need sampler
    #It then outputs a z value from a black-box distribution
    z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    #Note that if include_vars is True reconstructed will be a tuple of (means, log_vars)
    #Compute the discriminator value, which should represent the KL divergence if well-trained
    kl_loss = tf.reduce_mean(self.discriminator(inputs, z))
    reg_loss = self.regularizer(kl_loss)
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class CGModel(tf.keras.Model):
  """A special case of a VAE with a deterministic mapping for the encoder and a decoder
to invert the mapping. A flow is implemented on a standard normal prior so that new configs
can be generated without encoding. For a dimer, the CG coordinate is the distance between
dimer particles. For a lattice gas, the coordinate is the average density per site. To switch
between the two, 'system_type' should either be 'dimer' or 'lg'.
  """

  def __init__(self, data_shape, system_type,
               autoregress=False,
               num_latent=1, name='cgmodel', beta=1.0,
               use_skips=True,
               cg_map_info=None,
               **kwargs):
    super(CGModel, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.system = system_type
    self.autoregress = autoregress
    self.num_latent = num_latent
    self.beta = beta
    self.use_skips = use_skips
    self.cg_map_info = cg_map_info
    if self.system == 'dimer':
      self.encoder = architectures.DimerCGMapping()
      if self.autoregress:
        self.decoder = architectures.AutoregressiveDecoder(self.data_shape, return_vars=True, skip_connections=self.use_skips, auto_group_size=2)
      else:
        self.decoder = architectures.FCDecoder(self.data_shape, return_vars=True)
    elif self.system == 'lg':
      if self.cg_map_info is None:
        self.encoder = architectures.LatticeGasCGMapping()
      else:
        self.encoder = architectures.LatticeGasCGReduceMap(**cg_map_info)
      if self.autoregress:
        self.decoder = architectures.AutoregressiveDecoder(self.data_shape, skip_connections=self.use_skips)
      else:
        self.decoder = architectures.FCDecoder(self.data_shape)
    else:
      raise ValueError("System type of %s not understood."%self.system
                       +"\nMust be dimer or lg")
    #Because compressing to a latent dimension of 1, RealNVP can only scale and translate
    #To get most expressive flow, could use FFJORD, but it's slow, so use RQS
    flow_net_params = {'bin_range':[-10.0, 10.0], 'num_bins':32, 'hidden_dim':200}
    self.flow = architectures.NormFlowRQSplineRealNVP(self.num_latent,
                                                      kernel_initializer='truncated_normal',
                                                      rqs_params=flow_net_params)
    #If have encoder that doesn't go to 1D, need to flatten encoding
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs, training=False):
    z = self.encoder(inputs)
    z = self.flatten(z)
    #In this model, no KL divergence, but still want to maximize likelihood of P(z)
    #Here we define P(z) as a flow over a standard normal prior
    #So pass z through reverse flow to estimate likelihood
    #Should be able to do this AFTER training if like, but testing that idea out
    #If do after, MUST do really well before actually using model in MC simulations
    if self.beta != 0.0:
      #If regularization is zero, save time on the calculation
      z_prior, logdet = self.flow(z, reverse=True)
      #Estimate (negative) log likelihood of the prior
      logp_z = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(z_prior)
                                                + tf.math.log(2.0*math.pi),
                                                axis=1))
      #And SUBTRACT the average log determinant for the flow transformation
      logp_z -= tf.reduce_mean(logdet)
    else:
      logp_z = 0.0
    #Unlike with Srel model, want to model distribution of CG-space PARAMETERS with flow
    #That's why did flow first...
    #So should turn off sampling in reduction encoder so can manually sample here
    #If don't, sampling is just redundant (will return same configuration)
    #But only do if CG mapping is for lattice gas grid-size reduction
    if self.system == 'lg' and self.num_latent > 1:
      z = tf.cast((z > tf.random.uniform(z.shape)), dtype='float32')
    #With flow only on prior, z passes directly through
    if self.autoregress and training:
      reconstructed = self.decoder(z, train_data=inputs)
    else:
      reconstructed = self.decoder(z)
    reg_loss = self.beta*logp_z
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(logp_z), name='logp_z', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class SrelModel(tf.keras.Model):
  """A special case of a VAE with a deterministic mapping for the encoder and a decoder
to invert the mapping. The prior is a canonical ensemble distribution parametrized through
its potential energy function, which is modelled by cubic B-splines. This requires that
a separate optimization be performed for the decoder and the prior, as the parameters
and respective loss functions are not coupled in any way. Can choose between a dimer or
lattice gas system as 'dimer' or 'lg' input to the 'system_type' argument.

For the lattice gas, can also choose a different mapping, specifically a reduction
in the number of sites. By default this cg_map_info is None, in which case will stick to
encoding to average density. If do specify dictionary have the following options
{n_group:4, sample:True, sample_stochastic:True}.
  """

  def __init__(self, data_shape, system_type,
               autoregress=False,
               num_latent=1, name='srelmodel', beta=1.0,
               use_skips=True,
               cg_map_info=None,
               **kwargs):
    super(SrelModel, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.system = system_type
    self.autoregress = autoregress
    self.num_latent = num_latent
    self.beta = beta
    self.use_skips = use_skips
    self.cg_map_info = cg_map_info
    if self.system == 'dimer':
      self.encoder = architectures.DimerCGMapping()
      self.Ucg = architectures.SplinePotential(knot_points=np.linspace(0.8, 2.2, 40))
      if self.autoregress:
        self.decoder = architectures.AutoregressiveDecoder(self.data_shape, return_vars=True, skip_connections=self.use_skips, auto_group_size=2)
      else:
        self.decoder = architectures.FCDecoder(self.data_shape, return_vars=True)
    elif self.system == 'lg':
      if self.cg_map_info is None:
        self.encoder = architectures.LatticeGasCGMapping()
        self.Ucg = architectures.SplinePotential(knot_points=np.linspace(0.0, 1.0, 40))
      else:
        self.encoder = architectures.LatticeGasCGReduceMap(**cg_map_info)
        self.Ucg = architectures.ReducedLGPotential()
      if self.autoregress:
        self.decoder = architectures.AutoregressiveDecoder(self.data_shape, skip_connections=self.use_skips)
      else:
        self.decoder = architectures.FCDecoder(self.data_shape)
    else:
      raise ValueError("System type of %s not understood."%self.system
                       +"\nMust be dimer or lg")
    #If have encoder that doesn't go to 1D, need to flatten encoding
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs, training=False):
    z = self.encoder(inputs)
    #CG simulation is NOT a flow - if map from x0 to z0, want decoder to go back to x0
    #If allow CG simulation, may end up at z1 very different from z0
    #Then end up training decoder to map from z1 to x0... so it's NOT a flow
    #Only want to sample from CG ensemble if generating new configurations
    #And for any VAE class in this module, that happens outside the class
    #So get right on with decoding
    z = self.flatten(z)
    if self.autoregress and training:
      reconstructed = self.decoder(z, train_data=inputs)
    else:
      reconstructed = self.decoder(z)
    #In this model, no KL divergence, but still want to maximize likelihood of P(z)
    #This is equivalent to the Srel coarse-graining problem
    #With this type of problem cannot compute loss directly, but do know gradients
    #(because of difficulty in estimating partition function)
    #MUST do this in a separate training loop, then (parameters have no interdependence anyway)
    #To avoid breaking main training loop, just set things to zero
    logp_z = 0.0
    reg_loss = self.beta*logp_z
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(logp_z), name='logp_z', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class CGModel_AutoPrior(tf.keras.Model):
  """A special case of a VAE with a fixed mapping for the encoder and a flexible decoder
to invert the mapping. Instead of a flow, however, an autoregressive distribution for the
prior is learned, which can be used to generate new latent-space samples. As such, this
only makes sense if have multiple dimensions for num_latent, so only use it in such cases.
prior_params can be a dictionary of inputs to an AutoregressivePrior class.
Currently only intended for use with a lattice gas system based on the chosen mapping.
  """

  def __init__(self, data_shape,
               autoregress=False,
               num_latent=1, name='cgmodel_autoprior', beta=1.0,
               use_skips=True,
               cg_map_info={'n_group':4, 'sample':True, 'sample_stochastic':True},
               prior_params={'return_vars':False},
               **kwargs):
    super(CGModel_AutoPrior, self).__init__(name=name, **kwargs)
    self.data_shape = data_shape
    self.autoregress = autoregress
    self.num_latent = num_latent
    self.beta = beta
    self.use_skips = use_skips
    self.cg_map_info = cg_map_info
    self.prior_params = prior_params
    #Set up encoder
    self.encoder = architectures.LatticeGasCGReduceMap(**cg_map_info)
    if self.autoregress:
      self.decoder = architectures.AutoregressiveDecoder(self.data_shape, skip_connections=self.use_skips)
    else:
      self.decoder = architectures.FCDecoder(self.data_shape)
    #Set up autoregressive prior rather than flow
    latent_shape = tuple(np.ceil(np.array(self.data_shape)/4.0).astype(np.int32))
    self.prior = architectures.AutoregressivePrior(latent_shape, **prior_params)
    #If have encoder that doesn't go to 1D, need to flatten encoding
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs, training=False):
    z = self.encoder(inputs)
    #In this model, no KL divergence, but still want to maximize likelihood of P(z)
    #Here we define P(z) with an autoregressive model
    if self.beta != 0.0:
      #Estimate (negative) log likelihood of the prior
      logp_z = -tf.reduce_mean(self.prior(z.shape[0], train_data=z))
    else:
      logp_z = 0.0
    #With flow only on prior, z passes directly through
    z = self.flatten(z)
    if self.autoregress and training:
      reconstructed = self.decoder(z, train_data=inputs)
    else:
      reconstructed = self.decoder(z)
    reg_loss = self.beta*logp_z
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(logp_z), name='logp_z', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


class FullFlowVAE(tf.keras.Model):
  """VAE with normalizing flow on the prior AND a conditional flow on the posterior to
achieve maximal flexibility in the probability models. Fairly specialized with fewer options,
just because most options don't make sense here - throwing everything at the problem.
  """

  def __init__(self, data_shape, num_latent,
               name='fullflow_vae',
               beta=1.0,
               e_hidden_dim=1200,
               f_hidden_dim=200,
               d_hidden_dim=1200,
               n_auto_group=1,
               periodic_dof_inds=[],
               **kwargs):
    super(FullFlowVAE, self).__init__(name=name, **kwargs)
    self.autoregress = True #Set for compatibility
    self.sample_latent = True #Set for compatibility
    self.include_vars = True #Set for compatibility
    self.data_shape = data_shape
    self.num_latent = num_latent
    self.beta = beta
    self.e_hidden_dim = e_hidden_dim
    self.f_hidden_dim = f_hidden_dim
    self.d_hidden_dim = d_hidden_dim
    self.n_auto_group = n_auto_group
    #Allow periodic DOFs using VonMises distribution instead of Normal
    #This specifies their indices and builds boolean mask to pass to AutoregressiveDecoder
    self.periodic_dofs = [False,]*data_shape[0]
    for i in periodic_dof_inds:
      self.periodic_dofs[i] = True
    self.encoder = architectures.FCEncoder(num_latent, hidden_dim=self.e_hidden_dim,
                                           periodic_dofs=self.periodic_dofs)
    self.decoder = architectures.AutoregressiveFlowDecoder(data_shape,
                                                           hidden_dim=self.d_hidden_dim,
                                                           auto_group_size=self.n_auto_group,
                                                           periodic_dofs=self.periodic_dofs,
                                                           non_periodic_data_range=20.0)
    self.sampler = architectures.SampleLatent()
    flow_net_params = {'bin_range':[-10.0, 10.0], 'num_bins':32,
                       'hidden_dim':self.f_hidden_dim}
    self.flow = architectures.NormFlowRQSplineRealNVP(self.num_latent,
                                                      kernel_initializer='truncated_normal',
                                                      rqs_params=flow_net_params)

  def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
    del z_mean, z_logvar, z_sampled
    #For basic VAE, beta = 1.0, but want ability to change it
    return self.beta * kl_loss

  def call(self, inputs, training=False):
    z_mean, z_logvar = self.encoder(inputs)
    z = self.sampler(z_mean, z_logvar)
    #With flow only on prior, z passes directly through
    if training:
      reconstructed = self.decoder(z, train_data=inputs)
    else:
      reconstructed = self.decoder(z)
    #Before estimating KL divergence, pass z through inverse prior flow
    #(forward flow is used during generation from a standard normal)
    if self.beta != 0.0:
      z_prior, logdet = self.flow(z, reverse=True)
      #Estimate the KL divergence - should return average KL over batch
      kl_loss = losses.estimate_gaussian_kl(z_prior, z, z_mean, z_logvar)
      #And SUBTRACT the average log determinant for the flow transformation
      kl_loss -= tf.reduce_mean(logdet)
    else:
      kl_loss = 0.0
    reg_loss = self.regularizer(kl_loss, z_mean, z_logvar, z)
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(kl_loss), name='kl_loss', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    return reconstructed


