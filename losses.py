# Written by Jacob I. Monroe, NIST Employee

"""Library of loss functions for VAEs, including energies for physical systems.
Can also use these functions as metrics in keras models.
Adapted from disentanglement_lib https://github.com/google-research/disentanglement_lib"""
import copy
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

#Also for Mueller Potential
from deep_boltzmann.models.mueller_potential import MuellerPotential
_mueller_params = MuellerPotential.params_default.copy()
_mueller_params['k'] = 0.1
_muell_model = MuellerPotential(params=_mueller_params)

#For the polymer and alanine, need openmm for poential energies
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
import parmed as pmd
from openmmtools import testsystems


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


def estimate_gaussian_kl(tz, z, z_mean, z_logvar, prior_stds=None):
  """Estimates KL divergence by taking average over batch for latent space with a
normalizing flow. tz is the transformed coordinate, z the original, and z_mean and
z_logvar the mean and log variance of the original z. Will assume that the underlying
model for tz in the VAE framework is standard normal. Unless prior_stds are specified,
in which case the std of tz is specified for each sampled, allowing for explicitly
introducing temperature dependence into training.
  """
  if prior_stds is not None:
    prior_stds = tf.tile(tf.reshape(prior_stds, (tz.shape[0], 1)), (1, tz.shape[1]))
  else:
    prior_stds = tf.ones_like(tz)
  logp_z = -0.5*tf.reduce_sum(tf.square(z - z_mean)*tf.exp(-z_logvar)
                              + z_logvar,
                              #+ tf.math.log(2.0*np.pi),
                              axis=1)
  logp_tz = -0.5*tf.reduce_sum(tf.square(tz / prior_stds)
                               + 2.0*tf.math.log(prior_stds),
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


class openmmHamiltonian(object):
  """
  Base class for systems where need Hamiltonian from OpenMM (really just potential energy).
  Just to be used for inheritance as the __init__ method is not implemented. Each __init__
  should be unique to a particular molecular system.
  """

  def __init__(self):
    raise NotImplementedError("Must write __init__ for each unique system.")

  def __call__(self, configs, get_forces=False):
    configs = np.array(configs)
    pot_energies = np.zeros(configs.shape[0])
    forces = [] #Don't know shape because may transform configs
    for i, conf in enumerate(configs):
      if self.transform_func is not None:
        conf = self.transform_func(conf)
      #If configuration not finite, don't bother with energy to avoid OpenMM error
      #Would make sense to set to NaN, but that breaks training, so let be zero
      if not np.all(np.isfinite(conf)):
        #pot_energies[i] = np.nan
        if get_forces:
          forces.append(np.zeros_like(conf))
      else:
        self.sim.context.setPositions(conf*unit.angstroms)
        this_state = self.sim.context.getState(getEnergy=True, getForces=get_forces)
        pot_quantity = this_state.getPotentialEnergy()
        pot_energies[i] = pot_quantity.value_in_unit(unit.kilojoules_per_mole)
        if get_forces:
          #Only compute forces if energy finite
          if np.isfinite(pot_energies[i]):
            force_quantity = this_state.getForces()
            forces.append(force_quantity.value_in_unit(unit.kilojoules_per_mole/unit.angstroms))
          #Otherwise zero out so gradients zero
          else:
            forces.append(np.zeros_like(conf))
    if get_forces:
      return pot_energies, np.array(forces)
    else:
      return pot_energies
  

class polymerHamiltonian(openmmHamiltonian):

  def __init__(self, topFile, strucFile, transform_func=None):
    parmstruc = pmd.load_file(topFile, xyz=strucFile)
    self.top = parmstruc.topology
    self.system = parmstruc.createSystem(nonbondedMethod=app.NoCutoff,
                                         constraints=app.AllBonds,
                                         removeCMMotion=True,
                                         flexibleConstraints=False)
    self.sim = app.Simulation(self.top,
                              self.system,
                              mm.LangevinIntegrator(300.0*unit.kelvin,
                                                    1.0/unit.picoseconds,
                                                    2.0*unit.femtoseconds),
                              mm.Platform.getPlatformByName('CPU'))
    self.transform_func = transform_func
  

class dialanineHamiltonian(openmmHamiltonian):

  def __init__(self, transform_func=None):
    ala_vac = testsystems.AlanineDipeptideVacuum()
    ff = app.ForceField('amber99sb.xml')
    self.top = ala_vac.topology
    self.system = ff.createSystem(ala_vac.topology,
                                  nonbondedMethod=app.NoCutoff,
                                  constraints=app.HBonds)
    self.sim = app.Simulation(self.top,
                              self.system,
                              mm.LangevinIntegrator(300.0*unit.kelvin,
                                                    1.0/unit.picoseconds,
                                                    2.0*unit.femtoseconds),
                              mm.Platform.getPlatformByName('CPU'))
    self.transform_func = transform_func
  

def wrap_openmm_energy_tf(energy_func):
  """
  Given a function that computes configurational energies with OpenMM, which should take
  an optional argument get_forces to return forces, wraps this function with a
  @tf.custom_gradient so that the energy call can have gradients automatically computed
  (via the computed forces) by GradientTape.
  """
  @tf.custom_gradient
  def energy_with_grads(conf):
    conf = conf.numpy()
    energies, forces = energy_func(conf, get_forces=True)
    energies = tf.cast(energies, tf.float32)
    forces = tf.cast(forces, tf.float32)

    def _grad_fn(upstream):
      #Forces will always be (n_batch, n_atoms, 3)
      #That is because the input conf must also be this shape for OpenMM to work
      #So upstream has to be able to multiply that shape
      upstream = tf.reshape(upstream, (-1, 1, 1))
      grads = upstream * (-forces)
      return grads

    return energies, _grad_fn

  return energy_with_grads


def muellerHamiltonian(conf):
  """
  Wrapper around potential energy for Mueller Potential.
  """
  u = _muell_model.energy_tf(conf)
  return u


class WBPotential(object):
    """
    Defines the Washington Beltway Potential and its potential energies.
    """

    def __init__(self):
        self.offset = tf.cast(0.0, 'float32') #Offset between energy wells in radial distance, in kB*T
        self.mix = tf.cast(1.0 / (1.0 + np.exp(self.offset)), 'float32')
        self.r_dist = tfp.distributions.Mixture(tfp.distributions.Categorical(probs=[self.mix, 1.0-self.mix]),
                                               [tfp.distributions.Gamma(tf.cast(40.0, 'float32'), rate=tf.cast(40.0, 'float32')),
                                                tfp.distributions.Gamma(tf.cast(160.0, 'float32'), rate=tf.cast(80.0, 'float32'))])
        self.theta_dist = tfp.distributions.Uniform(low=tf.cast(-np.pi, 'float32'), high=tf.cast(np.pi, 'float32'))
        self.dist = tfp.distributions.JointDistributionSequential([self.r_dist, self.theta_dist])

    def energy_rtheta(self, vals):
        vals = tf.cast(vals, 'float32')
        pot = -self.dist.log_prob(tf.unstack(vals, axis=1))
        return pot

    def energy_xy(self, vals):
        r = np.sqrt(np.sum(vals**2, axis=-1))
        theta = np.arctan2(vals[:, 1], vals[:, 0])
        vals = np.vstack([r, theta]).T
        vals = tf.cast(vals, 'float32')
        pot = -self.dist.log_prob(tf.unstack(vals, axis=1))
        return pot


class ToyPotential(object):
    """
    Defines potentials for all toy models
    """
    @staticmethod
    def create_dist(f):
        #f is the probability to be in lower Gaussian peak
        if isinstance(f, (int, float)) or len(f.shape) == 0:
            ones_batch=tf.constant(1.0)
        else:
            f = tf.constant(f)
            ones_batch = tf.ones((f.shape[0],))
        probs = tf.stack([f, 1.0-f], axis=-1)
        return tfp.distributions.Mixture(cat=tfp.distributions.Categorical(probs=probs),
                                         components=[tfp.distributions.Normal(loc=-3.0*ones_batch, scale=ones_batch),
                                                     tfp.distributions.Normal(loc=3.0*ones_batch, scale=ones_batch)])

    def __init__(self, sys_type):
        if sys_type == 'ind_bi':
            self.dist = tfp.distributions.JointDistributionSequential([self.create_dist(0.6),
                                                                       self.create_dist(0.6)])
        elif sys_type == 'corr_bi':
            self.dist = tfp.distributions.JointDistributionSequential([self.create_dist(0.6),
                                                                       lambda x1: self.create_dist(-0.5*(tf.sign(x1)-1))])
        elif sys_type == 'ind_uni':
            self.dist = tfp.distributions.JointDistributionSequential([self.create_dist(1.0),
                                                                       self.create_dist(1.0)])
        else:
            raise ValueError('Unrecognized input for sys_type: %s\nSpecified system type must be string of: ind_bi, corr_bi, or ind_uni.'%str(sys_type))

    def energy(self, vals):
        vals = tf.cast(vals, 'float32')
        pot = -self.dist.log_prob(tf.unstack(vals, axis=1))
        return pot


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


def sampled_dimer_MSE_loss(true_confs, recon_info, sampler=gaussian_sampler):
  """Adds on sampling configuration from means and log variances to dimer energy calculation.
Then compares energies of samples (reconstructions) to those of actual configurations via MSE.
  """
  means = recon_info[0]
  logvars = recon_info[1]
  recon_confs = sampler(means, logvars)
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
  rand_log = np.log(np.random.random(conf.shape))
  rej = (log_prob < rand_log)
  new_conf[rej] = conf[rej]
  new_energy[rej] = energy[rej]
  return new_conf, new_energy


def sim_cg(init_conf, energy_func,
           num_steps=1, mc_noise=1.0, beta=1.0):
  """Simulates in the CG ensemble according to the provided energy function for the
specified number of steps.  A Gaussian MC move is utilized.  The final configuration and
energy is returned.
  """
  cg_confs = init_conf
  cg_energies = energy_func(cg_confs)
  for step in range(num_steps):
    cg_confs, cg_energies = gaussian_move(cg_confs, cg_energies, energy_func,
                                          noise_std=mc_noise, beta=beta)
  return cg_confs, cg_energies


#The below MC moves replicate code in mc_moves_LG.py only because that code imports losses
#for latticeGasHamiltonian, but then losses imports that code, so run into issues
def lgmc_moveTranslate(currConfig, currU, B,
                      energy_func=latticeGasHamiltonian,
                      energy_params={'mu':-2.0, 'eps':-1.0}):
  """Takes one particle and translates it to another random position.
     Computes the acceptance probability and returns it with the new
     configuration and potential energy.
  """
  #Get batch shape, and flattened indices of lattice sites
  batch_shape = currConfig.shape[0]
  site_inds = tf.range(np.prod(currConfig.shape[1:3]))[None, :]
  site_inds = tf.tile(site_inds, (batch_shape, 1))

  #Find where particles occupy lattice sites - flattened to access indices easier
  occupied = (currConfig == 1)
  occupied = tf.reshape(occupied, (batch_shape, np.prod(occupied.shape[1:])))
  unoccupied = (currConfig == 0)
  unoccupied = tf.reshape(unoccupied, (batch_shape, np.prod(unoccupied.shape[1:])))

  #Randomly select occupied and unoccupied indices to switch
  this_site = tf.map_fn(fn=tf.random.shuffle,
                        elems=tf.ragged.boolean_mask(site_inds, occupied))
  try:
    this_site = tf.squeeze(this_site[:, :1].to_tensor(default_value=-1), axis=-1)
  except tf.python.framework.errors_impl.InvalidArgumentError:
    #If above exception thrown, dimension 1 is 0, so cannot squeeze
    #In this case, just don't squeeze
    #Having a zero dimension means there aer no occupied sites
    #Would create problems when indexing, but will get masked out with batch_to_move
    this_site = this_site[:, :1].to_tensor(default_value=-1)
  new_site = tf.map_fn(fn=tf.random.shuffle,
                       elems=tf.ragged.boolean_mask(site_inds, unoccupied))
  try:
    new_site = tf.squeeze(new_site[:, :1].to_tensor(default_value=-1), axis=-1)
  except tf.python.framework.errors_impl.InvalidArgumentError:
    new_site = new_site[:, :1].to_tensor(default_value=-1)

  #If have no particles or vacancies, have zero probability of proposing the move
  #So move will be rejected
  batch_empty = (tf.reduce_sum(tf.cast(occupied, tf.int32), axis=1) == 0).numpy()
  batch_full = (tf.reduce_sum(tf.cast(unoccupied, tf.int32), axis=1) == 0).numpy()
  batch_to_move = tf.math.logical_and(tf.math.logical_not(batch_empty),
                                      tf.math.logical_not(batch_full)).numpy()

  #Create new configurations, flattened so can index sites to switch
  newConfig = np.reshape(copy.deepcopy(currConfig), (-1, np.prod(currConfig.shape[1:])))

  #For occupied sites, set to zero, for unoccupied, set to 1
  #But only for configurations that are not completely empty or full
  newConfig[batch_to_move, this_site[batch_to_move]] = 0
  newConfig[batch_to_move, new_site[batch_to_move]] = 1
  newConfig = np.reshape(newConfig, currConfig.shape)

  #Get new potential energy and compute acceptance probabilities
  newU = energy_func(newConfig, **energy_params).numpy()
  dU = newU - currU
  logPacc = -B*dU
  logPacc[batch_empty] = -np.inf
  logPacc[batch_full] = -np.inf

  return logPacc, newConfig, newU


def lgmc_moveDeleteMulti(currConfig, currU, B,
                        energy_func=latticeGasHamiltonian,
                        energy_params={'mu':-2.0, 'eps':-1.0}):
  """Takes a random number of particles and tries to delete.
     Returns the acceptance probability and new configuration and potential energy.
  """
  #Get batch shape, and flattened indices of lattice sites
  batch_shape = currConfig.shape[0]
  site_inds = tf.range(np.prod(currConfig.shape[1:3]))[None, :]
  site_inds = tf.tile(site_inds, (batch_shape, 1))

  #Find where particles occupy lattice sites - flattened to access indices easier
  occupied = (currConfig == 1)
  occupied = tf.reshape(occupied, (batch_shape, np.prod(occupied.shape[1:])))
  unoccupied = (currConfig == 0)
  unoccupied = tf.reshape(unoccupied, (batch_shape, np.prod(unoccupied.shape[1:])))

  #Need number of occupied and unoccupied
  num_oc = tf.reduce_sum(tf.cast(occupied, tf.float32), axis=1)
  num_un = tf.reduce_sum(tf.cast(unoccupied, tf.float32), axis=1)

  #And where batch has configs with at least one particle
  batch_to_remove = (num_oc > 0).numpy()

  #Randomly select number to delete and indices
  this_site = tf.map_fn(fn=tf.random.shuffle,
                        elems=tf.ragged.boolean_mask(site_inds, occupied))
  remove_lens = tf.math.minimum(this_site.row_lengths(), 20)

  #Cannot find way around looping, so just doing it to remove particles
  #Create new configurations and remove particles
  newConfig = np.reshape(copy.deepcopy(currConfig), (-1, np.prod(currConfig.shape[1:])))
  remove_num = []
  for i in range(batch_shape):
    if batch_to_remove[i]:
      this_remove_num = np.random.randint(1, remove_lens[i]+1)
      remove_num.append(this_remove_num)
      newConfig[i, this_site[i, :this_remove_num]] = 0

  newConfig = np.reshape(newConfig, currConfig.shape)
  remove_num = np.array(remove_num)

  #Get potential energy and calculate acceptance probabilities
  newU = energy_func(newConfig, **energy_params).numpy()
  logPacc = -B*(newU-currU)
  #If have no particles to delete, automatically reject
  logPacc[np.logical_not(batch_to_remove)] = -np.inf
  #Otherwise, add appropriate term for selecting indices
  logPacc[batch_to_remove] += tf.reduce_sum(tf.math.log(tf.ragged.range(num_oc[batch_to_remove]-remove_num, num_oc[batch_to_remove], dtype=tf.float32)+1), axis=1).numpy()
  logPacc[batch_to_remove] -= tf.reduce_sum(tf.math.log(tf.ragged.range(num_un[batch_to_remove], num_un[batch_to_remove]+remove_num, dtype=tf.float32)+1), axis=1).numpy()

  return logPacc, newConfig, newU


def lgmc_moveInsertMulti(currConfig, currU, B,
                        energy_func=latticeGasHamiltonian,
                        energy_params={'mu':-2.0, 'eps':-1.0}):
  """Takes a random number of particles and tries to delete.
     Returns the acceptance probability and new configuration and potential energy.
  """
  #Get batch shape, and flattened indices of lattice sites
  batch_shape = currConfig.shape[0]
  site_inds = tf.range(np.prod(currConfig.shape[1:3]))[None, :]
  site_inds = tf.tile(site_inds, (batch_shape, 1))

  #Find where particles occupy lattice sites - flattened to access indices easier
  occupied = (currConfig == 1)
  occupied = tf.reshape(occupied, (batch_shape, np.prod(occupied.shape[1:])))
  unoccupied = (currConfig == 0)
  unoccupied = tf.reshape(unoccupied, (batch_shape, np.prod(unoccupied.shape[1:])))

  #Need number of occupied and unoccupied
  num_oc = tf.reduce_sum(tf.cast(occupied, tf.float32), axis=1)
  num_un = tf.reduce_sum(tf.cast(unoccupied, tf.float32), axis=1)

  #And where batch has configs with at least one particle
  batch_to_insert = (num_un > 0).numpy()

  #Randomly select number to insert and indices
  this_site = tf.map_fn(fn=tf.random.shuffle,
                        elems=tf.ragged.boolean_mask(site_inds, unoccupied))
  insert_lens = tf.math.minimum(this_site.row_lengths(), 20)

  #Cannot find way around looping, so just doing it to insert particles
  #Create new configurations and insert particles
  newConfig = np.reshape(copy.deepcopy(currConfig), (-1, np.prod(currConfig.shape[1:])))
  insert_num = []
  for i in range(batch_shape):
    if batch_to_insert[i]:
      this_insert_num = np.random.randint(1, insert_lens[i]+1)
      insert_num.append(this_insert_num)
      newConfig[i, this_site[i, :this_insert_num]] = 1

  newConfig = np.reshape(newConfig, currConfig.shape)
  insert_num = np.array(insert_num)

  #Get potential energy and calculate acceptance probabilities
  newU = energy_func(newConfig, **energy_params).numpy()
  logPacc = -B*(newU-currU)
  #If have no particles to insert, automatically reject
  logPacc[np.logical_not(batch_to_insert)] = -np.inf
  #Otherwise, add appropriate term for selecting indices
  logPacc[batch_to_insert] += tf.reduce_sum(tf.math.log(tf.ragged.range(num_un[batch_to_insert]-insert_num, num_un[batch_to_insert], dtype=tf.float32)+1), axis=1).numpy()
  logPacc[batch_to_insert] -= tf.reduce_sum(tf.math.log(tf.ragged.range(num_oc[batch_to_insert], num_oc[batch_to_insert]+insert_num, dtype=tf.float32)+1), axis=1).numpy()

  return logPacc, newConfig, newU


def sim_reduced_lg_cg(init_conf, energy_func,
                      num_steps=1, move_probs=[0.5, 0.25, 0.25], beta=1.0):
  """Simulates specifically in the CG ensemble of a reduced lattice gas model (so just runs
an MC sim of a lattice gas). The move_probs list specifies probabilities of translation,
deletion, and insertion, in that order.
  """
  cg_confs = copy.deepcopy(init_conf)
  cg_energies = energy_func(cg_confs)
  if tf.is_tensor(cg_confs):
    cg_confs = cg_confs.numpy()
  if tf.is_tensor(cg_energies):
    cg_energies = cg_energies.numpy()
  for step in range(num_steps):
    move_type = np.random.choice(np.arange(3), p=move_probs)
    if move_type == 0:
      mc_move = lgmc_moveTranslate
    elif move_type == 1:
      mc_move = lgmc_moveDeleteMulti
    elif move_type == 2:
      mc_move = lgmc_moveInsertMulti
    logP, new_confs, new_energies = mc_move(cg_confs, cg_energies, beta,
                                            energy_func=energy_func, energy_params={})
    rand_logP = np.log(np.random.random(logP.shape[0]))
    to_acc = (logP > rand_logP)
    cg_confs[to_acc] = new_confs[to_acc]
    cg_energies[to_acc] = new_energies[to_acc]
  return cg_confs, cg_energies


def SrelLossGrad(confs, cg_pot,
                 cg_confs=None,
                 mc_sim=None,
                 mc_params={},
                 beta=1.0):
  """Calculates the gradients of the relative entropy loss of coarse-graining. The provided
configurations (confs) to average over should be in the coarse-grained coordinates. Note that
we cannot compute the loss directly without knowning the CG ensemble partition function, so
this only returns the gradients with respect to the coefficients.

If not in 1D, will switch default mc_sim and mc_params to simulate a lattice gas. This is
of course not general behavior, but prevents additional parameters in the training loop, which
is simpler for now since the only option in more than 1D is the 2D lattice gas.
  """
  if mc_sim is None:
    if len(confs.shape) > 2:
      #Only capture case for reduced lattice gas grid
      #To avoid this, must change training loop so specifies mc_sim and mc_params
      mc_sim = sim_reduced_lg_cg
      mc_params = {'num_steps':int(1e3), 'move_probs':[0.5, 0.25, 0.25], 'beta':beta}
    else:
      mc_sim = sim_cg
      mc_params = {'num_steps':int(1e3), 'mc_noise':0.1, 'beta':beta}
  #First term is average over full-resolution ensemble
  full_res_avg = np.average(cg_pot.get_coeff_derivs(confs), axis=0)
  #For next term need to average over CG ensemble, so run simulation in this ensemble first
  #That is, if cg_confs are not already provided
  if cg_confs is None:
    cg_confs, cg_energies = mc_sim(confs, cg_pot, **mc_params)
  cg_res_avg = np.average(cg_pot.get_coeff_derivs(cg_confs), axis=0)
  grads = beta*(full_res_avg - cg_res_avg)
  return grads


class AutoregressiveLoss(tf.keras.losses.Loss):
  """Given an autoregressive decoder, creates a loss function that will take in the decoder's
output and the true values and will return the log probability of the true values under the
decoders autoregressive probability distribution.
  """

  def __init__(self, decoder,
               name='auto_loss',
               **kwargs):
    super(AutoregressiveLoss, self).__init__(name=name, **kwargs)
    self.decoder = decoder

  def call(self, true_vals, recon_info):
    if hasattr(self.decoder, 'maf'):
      log_p = recon_info[-1]
    else:
      flat_true = self.decoder.flatten(true_vals)
      if self.decoder.return_vars:
        flat_recon = [self.decoder.flatten(info) for info in recon_info]
      else:
        flat_recon = self.decoder.flatten(recon_info)
      #If have periodic DOFs, need to unstack the DOFs before passing to log_prob
      if self.decoder.any_periodic:
        flat_true = tf.unstack(flat_true, axis=1)
      log_p = self.decoder.create_dist(flat_recon).log_prob(flat_true)
    return -log_p


class AutoConvLoss(tf.keras.losses.Loss):
  """Given a PixelCNN decoder, creates a loss function that will take logP of the PixelCNN's
probability distribution along with true values and just return the logP that got passed in.
Otherwise, replicate effort within the AutoConvDecoder layer. Just a hack so that the training
loop requires minimal alterations.
  """

  def __init__(self, decoder,
               name='auto_conv_loss',
               **kwargs):
    super(AutoConvLoss, self).__init__(name=name, **kwargs)
    self.decoder = decoder

  def call(self, true_vals, recon_info):
    log_p = recon_info
    return -log_p


class reverseKL_loss(object):
  """Computes an upper bound on the reverse KL divergence, so KL(P(x), Ptrue(x)). To do
this, you have to sample P(z), then sample P(x|z) and compute the potential energies of
generated configurations. And also average over P(x|z) for those same configs.
  """

  def __init__(self, vae, thermo_beta=1.0, potential=dimerHamiltonian):
    #Just need to define a VAE to use...
    self.vae = vae
    #Check if has a flow
    if hasattr(vae, 'flow'):
      self.do_flow = True
    else:
      self.do_flow = False
    #Set up quick way to compute log_p - expects loss, so will always take negative
    self.logp_func = AutoregressiveLoss(vae.decoder, reduction=tf.keras.losses.Reduction.NONE)
    #Need to set thermodynamic beta for defining dimensionless potential energy
    self.thermo_beta = tf.cast(thermo_beta, tf.float32)
    #Allows to modify potential energy function if need to, like for transformed coordinates
    self.potential = potential

  def reverse_gaussian_kl_clipped(self, tz, z, z_mean, z_logvar, min_logvar=-60.0):
    #Need to do in numerically stable way that prevents any infinities
    #Worry that logp_z will sometimes go to essentially negative infinity
    #This is just where the decoding -> encoding process makes the original z super unlikely
    #If get infinity then just clip, gradients will be NaN
    #So what we'll do is actually clip z_logvar
    #Then the computation of logp_z should be less than infinity if (z-z_mean)**2 not huge
    #BUT, the gradients will just be zero from that, not NaN
    # try:
    #     tf.debugging.assert_all_finite(z_logvar, 'Logvar not ok.')
    # except tf.errors.InvalidArgumentError as e:
    #     print('Had NaN or Inf in z_logvar, BEFORE clipping.', z_logvar)
    clipped_z_logvar = tf.where(z_logvar > min_logvar,
                                z_logvar,
                                min_logvar*tf.ones(tf.shape(z_logvar)))
    #For most stable computation, scale before squaring difference
    #Means need the std, so can go to smaller logvar because just need std to be non-zero
    clipped_std = tf.exp(0.5*clipped_z_logvar)
    logp_z = -0.5*tf.reduce_sum(tf.square(z/clipped_std - z_mean/clipped_std)
                                + clipped_z_logvar,
                                #+ tf.math.log(2.0*np.pi),
                                axis=1)
    # dist = tfp.distributions.Normal(z_mean, clipped_std)
    # logp_z = tf.reduce_sum(dist.log_prob(z), axis=1)
    # try:
    #     tf.debugging.assert_all_finite(logp_z, 'Log of q(z|x) not ok.')
    # except tf.errors.InvalidArgumentError as e:
    #     print('Had NaN or Inf in log of q(z|x).', logp_z)
    logp_tz = -0.5*tf.reduce_sum(tf.square(tz),
                                 #+ tf.math.log(2.0*np.pi),
                                 axis=1)
    return tf.reduce_mean(logp_tz - logp_z)

  def __call__(self,
               x_train,
               unused_recon_info):
    #Remove unused information
    del unused_recon_info

    #Get batch shape
    batch_size = tf.shape(x_train)[0]
    #First draw a z sample from a standard normal (assumes this is form of latent)
    tz_sample = tf.random.normal((batch_size, self.vae.num_latent))

    #Pass through flow if have one
    if self.do_flow:
      z_sample, logdet = self.vae.flow(tz_sample, reverse=False)
    else:
      z_sample = tz_sample
      logdet = 0.0

    #Pass through decoder and draw samples
    #Will be slow if autoregressive
    x_info = self.vae.decoder(z_sample)

    #By calling without training data, will actually sample P(x|z)
    #If using tfp, will do this in way that is differentiable - just be careful
    x_sample = x_info[-1]
    x_info = x_info[:-1]

    #And get probabilities of all configurations
    #logp_func is going to (almost always) be specified as a loss, so negate
    log_Pxz = -self.logp_func(x_sample, x_info)

    #And get all potential energies - must be done with tf functions compatible with autograd!
    u_pot = tf.cast(self.potential(x_sample), 'float32')

    #Finally, need to encode generated sample to obtain q(z|x)
    z_means, z_logvars = self.vae.encoder(x_sample)

    #And compute the pseudo KL term (not true KL divergence) to ensure q(z|x) close to P(z)
    log_Pz_over_qzx = self.reverse_gaussian_kl_clipped(tz_sample, z_sample, z_means, z_logvars)
    log_Pz_over_qzx -= tf.reduce_mean(logdet)

    loss = tf.reduce_mean(self.thermo_beta*u_pot + log_Pxz) + log_Pz_over_qzx
    return loss

