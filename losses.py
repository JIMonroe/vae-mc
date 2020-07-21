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

"""Library of basic reconstruction and ELBO loss functions for VAEs.
Can also use these functions as metrics in keras models.
Adapted from disentanglement_lib https://github.com/google-research/disentanglement_lib"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

#For the dimer system, nice to have the energy function easily on hand
#Not perfectly efficient or pythonian, but take care of that here, creating global definitions
#However, will try and keep these definitions within this module by indicating privacy
from deep_boltzmann.models.particle_dimer import ParticleDimer
_dimer_params = ParticleDimer.params_default.copy()
_dimer_params['dimer_slope'] = 2.0
_dim_model = ParticleDimer(params=_dimer_params)


#Identical to tf.keras.losses.BinaryCrossentropy(from_logits=True)
#(if activation is "logits")
#BUT, have to somehow get that loss function to do reduce_sum on all but first dimension
#To technically be the same...
#If set reduction in loss to be AUTO, get output of this divided by 4096
#If set to SUM, get output of this multiplied by the batch size (64)
#Then the KL divergence loss gets added to one of these
#So really just changes the lambda scaling that we want to use
def bernoulli_loss(true_images,
                   reconstructed_images,
                   activation="logits",
                   subtract_true_image_entropy=False):
  """Computes the Bernoulli loss."""
  #Can't seem to access shape of true_images...
  #Something weird with how the iterator works
  #But, don't need to flatten if just change axes used with reduce_sum
  #flattened_dim = np.prod(true_images.shape[1:])
  #reconstructed_images = tf.reshape(
  #    reconstructed_images, shape=[-1, flattened_dim])
  #true_images = tf.reshape(true_images, shape=[-1, flattened_dim])

  # Because true images are not binary, the lower bound in the xent is not zero:
  # the lower bound in the xent is the entropy of the true images.
  if subtract_true_image_entropy:
    dist = tfp.distributions.Bernoulli(
        probs=tf.clip_by_value(true_images, 1e-6, 1 - 1e-6))
    loss_lower_bound = tf.reduce_sum(dist.entropy(), axis=[1,2,3])
  else:
    loss_lower_bound = 0

  if activation == "logits":
    loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=reconstructed_images, labels=true_images),
        axis=[1,2,3])
  elif activation == "tanh":
    reconstructed_images = tf.clip_by_value(
        tf.nn.tanh(reconstructed_images) / 2 + 0.5, 1e-6, 1 - 1e-6)
    loss = -tf.reduce_sum(
        true_images * tf.math.log(reconstructed_images) +
        (1 - true_images) * tf.math.log(1 - reconstructed_images),
        axis=[1,2,3])
  else:
    raise NotImplementedError("Activation not supported.")

  return loss - loss_lower_bound


#Generic loss that would work for particle-based simulation or images/lattices
#Essentially just a generalization of MSE loss
#Does assume a diagonal covariance matrix
def diag_gaussian_loss(true_vals,
                       recon_info,
                       activation=None):
  """Computes the loss assuming a Gaussian distribution (with diagonal covariance matrix) for
the probability distribution of the reconstruction model."""
  recon_means = recon_info[0]
  recon_log_var = recon_info[1]
  #Negative logP is represented below
  mse_term = 0.5*tf.square(true_vals - recon_means)*tf.exp(-recon_log_var)
  reg_term = 0.5*recon_log_var
  #norm_term = 0.5*tf.math.log(2.0*np.pi)
  sum_terms = tf.reduce_sum(mse_term + reg_term, # + norm_term,
                            axis=np.arange(1, len(true_vals.shape)))
  loss = sum_terms #tf.reduce_sum(sum_terms) #Summing loss over all samples to return
  #Instead return per sample loss... use ReconLoss and reduction to set to sum or take mean
  return loss


#Identical to tf.keras.losses.mse(), but sigmoid applied to reconstruction first
#(if activation is "logits")
#AND, have to somehow get that loss function to do reduce_sum on all but first dimension
def l2_loss(true_images, reconstructed_images, activation="logits"):
  """Computes the l2 loss."""
  if activation == "logits":
    return tf.reduce_sum(
        tf.square(true_images - tf.nn.sigmoid(reconstructed_images)), axis=[1, 2, 3])
  elif activation == "tanh":
    reconstructed_images = tf.nn.tanh(reconstructed_images) / 2 + 0.5
    return tf.reduce_sum(
        tf.square(true_images - reconstructed_images), axis=[1, 2, 3])
  else:
    raise NotImplementedError("Activation not supported.")


def compute_gaussian_kl(z_mean, z_logvar):
  """Compute KL divergence between input Gaussian and Standard Normal."""
  #Can return per-sample KL divergence to be consistent with reconstruction loss
  #When take mean of reconstruction loss and this, you get the correct result
  #If regularizer scales the KL divergence or adds constant to it, still works out
  per_sample_kl = (0.5 * tf.reduce_sum(
                   tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1])
                  )
  return tf.reduce_mean(per_sample_kl)


def estimate_gaussian_kl(tz, z, z_mean, z_logvar):
  """Estimates KL divergence by taking average over batch for latent space with a
normalizing flow. tz is the transformed coordinate, z the original, and z_mean and
z_logvar the mean and log variance of the original z. Will assume that the underlying
model for tz in the VAE framework is standard normal.
  """
  logp_z = -0.5*tf.reduce_sum(tf.square(z - z_mean)*tf.exp(-z_logvar)
                              + z_logvar,
                              #+ tf.math.log(2.0*np.pi),
                              axis=1)
  logp_tz = -0.5*tf.reduce_sum(tf.square(tz),
                               #+ tf.math.log(2.0*np.pi),
                               axis=1)
  return tf.reduce_mean(logp_z - logp_tz)


def reconstruction_loss(loss_fn=bernoulli_loss,
                        activation="logits"):
  """Wrapper that creates reconstruction loss."""

  #Will actually return a function that takes two inputs, the true and reconstructed images
  #But the function can be customized by specifying different loss_fn and activation
  def calc_loss(true_images, reconstructed_images):
    per_sample_loss = loss_fn(true_images, reconstructed_images, activation)
    return per_sample_loss

  return calc_loss


def latticeGasHamiltonian(conf, mu=-2.0, eps=-1.0):
  """Computes the Hamiltonian for a 2D lattice gas given an image - so if the
image is 1's and 0's it makes sense. Really should be any number of images. If
just have one, add a first dimension to it.
  """
  tfconf = tf.cast(conf, 'float32')
  #Shift all indices by 1 in up and down then left and right and multiply by original
  ud = eps*tfconf*tf.roll(tfconf, 1, axis=1)
  lr = eps*tfconf*tf.roll(tfconf, 1, axis=2)
  #Next incorporate chemical potential term
  chempot = mu*tfconf
  #And sum everything up
  H = tf.reduce_sum(ud + lr - chempot, axis=np.arange(1, len(tfconf.shape)))
  return H


def linlogcut_tf(x, high_E=100, max_E=1e10):
  """Function to clip large energies - taken from Frank Noe's deep_boltzmann package.
  """
  # cutoff x after max_E - this should also cutoff infinities
  x = tf.where(x < max_E, x, max_E * tf.ones(tf.shape(x)))
  # log after high_E
  y = high_E + tf.where(x < high_E, x - high_E, tf.math.log(x - high_E + 1))
  # make sure everything is finite
  y = tf.where(tf.math.is_finite(y), y, max_E * tf.ones(tf.shape(y)))
  return y


def dimerHamiltonian(conf, doClip=True):
  u = _dim_model.energy_tf(conf)
  if doClip:
    return linlogcut_tf(u, high_E=10000, max_E=1e10)
  else:
    return u


def gaussian_sampler(mean, logvar):
  """Simple samples a Gaussian distribution given means and log variances. Same as using
the reparametrization trick so can add this to a loss function and use samples rather than
means to do things like compute energies.
  """
  return tf.add(mean,
                tf.exp(logvar / 2.0) * tf.random.normal(tf.shape(mean), 0, 1)
               )


def binary_sampler(logits):
  """Samples EXACTLY from a Bernoulli distribution rather than approximately. The gradient
of this operation cannot be computed, so it will not contribute to a loss! This is only
defined for convenience of drawing lattice gas configurations.
  """
  probs = tf.math.sigmoid(logits)
  rand_vals = tf.random.uniform(probs.shape)
  return tf.cast((probs > rand_vals), dtype='float32')


def bernoulli_sampler(logits, beta=100.0):
  """Implements the reparametrization trick but (approximately) for a Bernoulli distribution
(see Maddison, Mnih, and Teh, "The concrete distribution: A continuous relaxation fo discrete
random variables," 2016 for more information. Essentially takes a logit (NOT a probability)
and converting the output into a zero or one by drawing another random number from a uniform
distribution. The larger beta is, the closer to zero or one the output will be, but you pay
the price of having sharper gradients in your optimization.
  """
  eps = tf.random.uniform(tf.shape(logits))
  return tf.math.sigmoid(beta*(logits + tf.math.log(eps) - tf.math.log(1.0-eps)))


def sampled_dimer_MSE_loss(true_confs, recon_info):
  """Adds on sampling configuration from means and log variances to dimer energy calculation.
Then compares energies of samples (reconstructions) to those of actual configurations via MSE.
  """
  means = recon_info[0]
  logvars = recon_info[1]
  recon_confs = gaussian_sampler(means, logvars)
  recon_energy = dimerHamiltonian(recon_confs)
  true_energy = dimerHamiltonian(true_confs)
  return tf.keras.losses.mse(true_energy, recon_energy)


def sampled_lg_MSE_loss(true_confs, logits, sample_beta=100.0,
                        energy_params={'mu':-2.0, 'eps':-1.0}):
  """Samples (approximately) Bernoulli variables (zeros and ones) from given logits, then
calculates energy and compares to energy of true configurations via MSE.
  """
  recon_confs = bernoulli_sampler(logits, beta=sample_beta)
  recon_energy = latticeGasHamiltonian(recon_confs, **energy_params)
  true_energy = latticeGasHamiltonian(true_confs, **energy_params)
  return tf.keras.losses.mse(true_energy, recon_energy)


#Below function is very general, but not as convenient as the ones specific for dimer or LG
#Those actually allow fluctuations to be included through sampling under the reparametrization
#trick - in other words, the energy is handled differently for the true configs and recons
def transform_MSE_loss(transform_fn=latticeGasHamiltonian,
                       func_params=[-2.0, -1.0],
                       activation=None,
                       weightFactor=1.0):
  """Wrapper to create a loss function returning the MSE between transformations of
configurations. Useful for adding on loss for potential energies, etc. The passed
function must compute the transform for each sample (i.e. doesn't combine along
the first axis)."""

  def calc_loss(true_images, recon_images):
    if activation is "logits":
      recon_images = tf.math.sigmoid(recon_images)
    if isinstance(func_params, dict):
      true_transform = transform_fn(true_images, **func_params)
      recon_transform = transform_fn(recon_images, **func_params)
    else:
      true_transform = transform_fn(true_images, *func_params)
      recon_transform = transform_fn(recon_images, *func_params)
    return weightFactor*tf.keras.losses.mse(true_transform, recon_transform)

  return calc_loss


def relative_boltzmann_loss(x_sample,
                            x_probs,
                            energy_func=latticeGasHamiltonian,
                            func_params=[-2.0, -1.0],
                            beta=1.0):
  """Loss function that computes the MSE between relative probability weights predicted
by a vae_model and those known to occur for a Boltzmann distribution. In other words, the
relative Boltzmann log weights are the reduced potential differences and the relative log
weights under the model are the differences between the log of the expectation of P(x|z)
over z. A vae_model must be provided so that we can compute P(x|z). No examples are
necessary to train this objective function as the relative Boltzmann weights are known
a priori.
  """
  #Get all potential energies now (before fancy reshaping, etc.)
  u_pot = energy_func(x_sample, *func_params)

  #Can use broadcasting trick to do everything in one shot, no loop (at least not in python)
  #Requires more memory - specifically multiplies size of x_sample by x_sample.shape[0]
  x_sample = tf.reshape(x_sample, (x_sample.shape[0],1)+x_sample.shape[1:])
  log_pxz = tf.reduce_sum(x_sample*tf.math.log(x_probs)
                          + (1.0 - x_sample)*tf.math.log(1.0 - x_probs), axis=(2,3,4))
  log_avg_pxz = tf.reduce_logsumexp(log_pxz, axis=(1)) - tf.math.log(float(x_probs.shape[0]))

  loss_px = tf.reduce_mean(log_avg_pxz)
  loss_u = tf.reduce_mean(beta*u_pot)

  loss = loss_px + loss_u

  #And compute loss using first configuration as reference
  #loss = tf.reduce_mean(tf.math.square(log_avg_pxz[1:] - log_avg_pxz[0]
  #                                     + beta*(u_pot[1:] - u_pot[0])))
  #loss = tf.reduce_mean(log_avg_pxz + beta*u_pot)
  return loss, loss_px, loss_u


class ReconLoss(tf.keras.losses.Loss):
  """Computes just the reconstruction loss."""

  def __init__(self,
               name='recon_loss',
               loss_fn=bernoulli_loss, activation="logits", 
               **kwargs):
    super(ReconLoss, self).__init__(name=name, **kwargs)
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)

  def call(self, true_images, reconstructed_images):
    #Note that reconstruction_loss returns per_sample loss
    #But calling ReconLoss will return the reduced mean of the per sample loss
    #This is the default behavior of the tf.keras.losss.Loss class
    return self.recon_loss(true_images, reconstructed_images)


#It's probably NOT a good idea to use this function for the loss passed to the keras model
#To create this loss, you have to provide a VAE model class
#Then that class gets evaluated in addition to during the optimization to compute this loss
#So using this slows things down, but also make the code more confusing
#Instead, use some reconstruction loss of your choice
#The regularization is added in the class itself
class TotalVAELoss(tf.keras.losses.Loss):
  """Computes the total loss, reconstruction and regularizer based on KL divergence."""

  def __init__(self, vae_model,
               name='vae_loss',
               loss_fn=bernoulli_loss, activation="logits", 
               **kwargs):
    full_name = '%s_%s'%(name, vae_model.name)
    super(TotalVAELoss, self).__init__(name=full_name, **kwargs)
    self.vae_model = vae_model
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)

  #Directly using a VAE model object to define a loss function of two inputs only
  def call(self, true_images, reconstructed_images):
    z_mean, z_logvar = self.vae_model.encoder(true_images)
    z_sampled = self.vae_model.sampler(z_mean, z_logvar)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    return tf.add(self.recon_loss(true_images, reconstructed_images),
                  self.vae_model.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
                 )


class ReconLossMetric(tf.keras.metrics.Metric):
  """Computes the reconstruction loss as a metric class object."""

  def __init__(self, name='reconstruction_loss', 
               loss_fn=bernoulli_loss, activation="logits", 
               **kwargs):
    super(ReconLossMetric, self).__init__(name=name, **kwargs)
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)
    self.current_loss = self.add_weight(name='curr_recon_loss',
                                        initializer='zeros')

  def update_state(self, true_images, reconstructed_images, sample_weight=None):
    values = self.recon_loss(true_images, reconstructed_images)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      values = tf.multiply(values, sample_weight)
      curr_sum = tf.reduce_sum(values) / tf.reduce_sum(sample_weight)
    else:
      curr_sum = tf.reduce_mean(values)
    #Most metrics accumulate - here just replace the value instead of running average
    self.current_loss.assign(curr_sum)

  def result(self):
    return self.current_loss

  def reset_states(self):
    return self.current_loss.assign(0.)


class ElboLossMetric(tf.keras.metrics.Metric):
  """Computes ELBO as sum of reconstruction loss and KL divergence of Gaussians."""

  def __init__(self, vae_model,
               name='ELBO_loss',
               loss_fn=bernoulli_loss, activation="logits",
               **kwargs):
    super(ElboLossMetric, self).__init__(name=name, **kwargs)
    self.vae_model = vae_model
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)
    self.current_elbo = self.add_weight(name='curr_ELBO_loss',
                                        initializer='zeros')

  def update_state(self, true_images, reconstructed_images, sample_weight=None):
    #The overall loss function is tied directly to a VAE model object
    #It uses the encoder of the VAE to get the latent spaces means and variances
    z_mean, z_logvar = self.vae_model.encoder(true_images)
    values = tf.add(self.recon_loss(true_images, reconstructed_images),
                    compute_gaussian_kl(z_mean, z_logvar)
                   )
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      values = tf.multiply(values, sample_weight)
      curr_sum = tf.reduce_sum(values) / tf.reduce_sum(sample_weight)
    else:
      curr_sum = tf.reduce_mean(values)
    #Most metrics accumulate - here just replace the value instead of running average
    self.current_elbo.assign(curr_sum)

  def result(self):
    return self.current_elbo

  def reset_states(self):
    return self.current_elbo.assign(0.)


class TotalLossMetric(tf.keras.metrics.Metric):
  """Computes the total loss, reconstruction and regularizer based on KL divergence."""

  def __init__(self, vae_model,
               name='total_loss',
               loss_fn=bernoulli_loss, activation="logits", 
               **kwargs):
    super(TotalLossMetric, self).__init__(name=name, **kwargs)
    self.vae_model = vae_model
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)
    self.current_total = self.add_weight(name='curr_total_loss',
                                         initializer='zeros')

  def update_state(self, true_images, reconstructed_images, sample_weight=None):
    #Directly using a VAE model object to define a loss function of two inputs only
    z_mean, z_logvar = self.vae_model.encoder(true_images)
    z_sampled = self.vae_model.sampler(z_mean, z_logvar)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    values = tf.add(self.recon_loss(true_images, reconstructed_images),
                    self.vae_model.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
                   )
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      values = tf.multiply(values, sample_weight)
      curr_sum = tf.reduce_sum(values) / tf.reduce_sum(sample_weight)
    else:
      curr_sum = tf.reduce_mean(values)
    #Don't accumulate this metric, just replace current value
    self.current_total.assign(curr_sum)

  def result(self):
    return self.current_total

  def reset_states(self):
    return self.current_total.assign(0.)


class RegularizerLossMetric(tf.keras.metrics.Metric):
  """Computes just the regularizer loss based on KL divergence."""

  def __init__(self, vae_model,
               name='regularizer_loss',
               loss_fn=bernoulli_loss, activation="logits", 
               **kwargs):
    super(RegularizerLossMetric, self).__init__(name=name, **kwargs)
    self.vae_model = vae_model
    self.recon_loss = reconstruction_loss(loss_fn=loss_fn, activation=activation)
    self.current_reg = self.add_weight(name='curr_reg_loss',
                                       initializer='zeros')

  def update_state(self, true_images, reconstructed_images, sample_weight=None):
    #Directly using a VAE model object to define a loss function of two inputs only
    z_mean, z_logvar = self.vae_model.encoder(true_images)
    z_sampled = self.vae_model.sampler(z_mean, z_logvar)
    kl_loss = compute_gaussian_kl(z_mean, z_logvar)
    values = self.vae_model.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      values = tf.multiply(values, sample_weight)
      curr_sum = tf.reduce_sum(values) / tf.reduce_sum(sample_weight)
    else:
      curr_sum = tf.reduce_mean(values)
    #Don't accumulate this metric, just replace current value
    self.current_reg.assign(curr_sum)

  def result(self):
    return self.current_reg

  def reset_states(self):
    return self.current_reg.assign(0.)


def gaussian_move(conf, energy, energy_func,
                  noise_std=1.0, beta=1.0):
  """Uses Gaussian noise to generate an MC move given a configuration. Returns the
new configuration if accepted and the old if rejected based on Metropolis criteria.
Works for arbitrary numbers of configurations.
  """
  noise = np.random.normal(loc=0.0, scale=noise_std, size=conf.shape)
  new_conf = conf + noise
  new_energy = energy_func(new_conf)
  #For Gaussian noise, probability of reverse move is same as forward due to symmetry
  log_prob = -beta*(new_energy - energy)
  rand_log = np.log(np.random.random(conf.shape[0]))
  rej = np.where(log_prob < rand_log)[0]
  new_conf[rej] = conf[rej]
  new_energy[rej] = energy[rej]
  return new_conf, new_energy


def SrelLossGrad(confs, mc_move_func, cg_pot,
                 num_steps=int(1e3), mc_noise=1.0, beta=1.0):
  """Calculates the gradients of the relative entropy loss of coarse-graining. The provided
configurations (confs) to average over should be in the coarse-grained coordinates. Note that
we cannot compute the loss directly without knowning the CG ensemble partition function, so
this only returns the gradients with respect to the coefficients.
  """
  #import matplotlib.pyplot as plt
  #First term is average over full-resolution ensemble
  full_res_avg = np.average(cg_pot.get_coeff_derivs(confs), axis=0)
  #For next term need to average over CG ensemble, so run simulation in this ensemble first
  cg_confs = confs
  cg_energies = cg_pot(cg_confs)
  #to_plot = np.zeros(num_steps+1)
  #to_plot[0] = cg_confs[0]
  for step in range(num_steps):
    cg_confs, cg_energies = mc_move_func(cg_confs, cg_energies, cg_pot,
                                         noise_std=mc_noise, beta=beta)
    #to_plot[step+1] = cg_confs[0]
  #plt.plot(cg_confs)
  #plt.show()
  cg_res_avg = np.average(cg_pot.get_coeff_derivs(cg_confs), axis=0)
  grads = beta*(full_res_avg - cg_res_avg)
  return grads


