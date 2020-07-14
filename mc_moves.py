
"""Defines Monte Carlo moves for VAEs and invertible networks.
Useful for testing MC efficiency without actually running MC simulations.
Keeping moves general so will work for any system (hopefully).
"""

import copy
import numpy as np
import tensorflow as tf
from libVAE import vae, dataloaders, losses


def zDrawNormal(mean, logvar, zSel=None, nDraws=1):
  """Directly draws z from a normal distribution based on mean and logvar
  """
  if zSel is not None:
    if nDraws == 1:
      zval = zSel
    else:
      zval = tf.random.normal((nDraws-1, mean.shape[0]),
                              mean=mean, stddev=tf.math.exp(0.5*logvar))
      zval = tf.concat((zSel, zval), 0)
  else:
      zval = tf.random.normal((nDraws, mean.shape[0]),
                              mean=mean, stddev=tf.math.exp(0.5*logvar))
  zlogprob = tf.reduce_sum( -0.5*tf.math.square(zval - mean)/tf.math.exp(logvar)
                            - 0.5*np.log(2.0*np.pi) - 0.5*logvar,  axis=1).numpy()
  return zval, zlogprob


def zDrawUniform(minZ, maxZ, zSel=None, nDraws=1):
  """Directly draws from a uniform distribution based on given min and max
  """
  if zSel is not None:
    if nDraws == 1:
      zval = zSel
    else:
      zval = tf.random.uniform((nDraws-1, minZ.shape[0]), minval=minZ, maxval=maxZ)
      zval = tf.concat((zSel, zval), 0)
  else:
    zval = tf.random.uniform((nDraws, minZ.shape[0]), minval=minZ, maxval=maxZ)
  zlogprob = tf.math.reduce_sum(-tf.math.log(maxZ - minZ)).numpy()
  zlogprob = zlogprob*tf.ones(nDraws).numpy()
  return zval, zlogprob


def zDrawDirect(dat_means, dat_logvars, zSel=None, nDraws=1, flow=None):
  """Draws a z value from a sample of z means and log variances output by the encoder.
Of course assuming that the sampling is from a normal distribution.
  """
  if zSel is not None:
    if flow is not None:
      zSel, sel_log_det_rev = flow(zSel, reverse=True)
    if nDraws == 1:
      zval = zSel
    else:
      randInd = np.random.choice(dat_means.shape[0], size=nDraws-1)
      randMean = tf.gather(dat_means, randInd)
      randLogVar = tf.gather(dat_logvars, randInd)
      zval = tf.random.normal(randMean.shape,
                              mean=randMean, stddev=tf.math.exp(0.5*randLogVar))
      zval = tf.concat((zSel, zval), 0)
  else:
    randInd = np.random.choice(dat_means.shape[0], size=nDraws)
    randMean = tf.gather(dat_means, randInd)
    randLogVar = tf.gather(dat_logvars, randInd)
    zval = tf.random.normal(randMean.shape,
                            mean=randMean, stddev=tf.math.exp(0.5*randLogVar))
  #Probability of this z value is average of P(z|x) over all x, which is sum of Gaussians
  #To make broadcasting work, reshape zval
  zvalCalc = tf.reshape(zval, (zval.shape[0], 1, zval.shape[1]))
  zlogprob = tf.reduce_logsumexp(-0.5*tf.math.square(zvalCalc-dat_means)/tf.math.exp(dat_logvars)
                                 - 0.5*np.log(2.0*np.pi) - 0.5*dat_logvars, axis=1)
  zlogprob = zlogprob - tf.math.log(tf.cast(dat_means.shape[0], 'float32'))
  zlogprob = tf.reduce_sum(zlogprob, axis=1).numpy()
  if flow is not None:
    zval, log_det = flow(zval)
    zlogprob -= log_det.numpy()
  return zval, zlogprob


class zDrawFunc_wrap_VAE(object):
  """Based on data, wraps all functions to draw z values. Allows to precompute statistics
given data so that don't have to redo every MC step. Also need to provide a VAE model for
interpreting the data.
  """
  def __init__(self, dat, vae_model):
    self.zMeans, self.zLogvars = vae_model.encoder(dat)
    self.trueMean = tf.reduce_mean(self.zMeans, axis=0)
    self.trueLogVar = tf.math.log(tf.reduce_mean(tf.math.exp(self.zLogvars)
                                                 + tf.square(self.zMeans)
                                                 - tf.square(self.trueMean), axis=0))
    self.minZ = self.trueMean-5*tf.exp(0.5*self.trueLogVar)
    self.maxZ = self.trueMean+5*tf.exp(0.5*self.trueLogVar)
    try:
      self.flow = vae_model.flow
    except AttributeError:
      self.flow = None

  def __call__(self, draw_type='std_normal', zSel=None, nDraws=1):
    if draw_type == 'std_normal':
      return  zDrawNormal(tf.zeros(self.trueMean.shape),
                          tf.zeros(self.trueLogVar.shape),
                          zSel=zSel, nDraws=nDraws)
    elif draw_type == 'normal':
      return  zDrawNormal(self.trueMean, self.trueLogVar, zSel=zSel, nDraws=nDraws)
    elif draw_type == 'uniform':
      return zDrawUniform(self.minZ, self.maxZ, zSel=zSel, nDraws=nDraws)
    elif draw_type == 'direct':
        return zDrawDirect(self.zMeans, self.zLogvars,
                           zSel=zSel, nDraws=nDraws, flow=self.flow)
    else:
      print('Draw style unknown.')
      return None


def zDrawLocal(vae_model, x, zSel=None, nDraws=1):
  """Draws latent space coordinates given real-space coordinates. This constitutes
a local move, increasing the acceptance probability. Can instead select a z value
and calculate its probability given x.
  """
  #Generate the mean and log variance of the normal distribution for z given x
  zMean, zLogvar = vae_model.encoder(tf.reshape(tf.cast(x, 'float32'), (1,)+x.shape+(1,)))
  if zSel is not None:
    try:
      zSel, sel_log_det_rev = vae_model.flow(zSel, reverse=True)
    except AttributeError:
      pass
    if nDraws == 1:
      zval = zSel
    else:
      zval = vae_model.sampler(tf.tile(zMean, (nDraws-1,1)), tf.tile(zLogvar, (nDraws-1,1)))
      zval = tf.concat((zSel, zval), 0)
  else:
    zval = vae_model.sampler(tf.tile(zMean, (nDraws,1)), tf.tile(zLogvar, (nDraws,1)))
  zlogprob = tf.reduce_sum( -0.5*tf.math.square(zval - zMean)/tf.math.exp(zLogvar)
                            - 0.5*np.log(2.0*np.pi) - 0.5*zLogvar, axis=1).numpy()
  try:
    zval, log_det = vae_model.flow(zval)
    zlogprob -= log_det.numpy()
  except AttributeError:
    pass
  return zval, zlogprob


def xDrawSimple(vae_model, z, xSel=None, activation=tf.math.sigmoid,
                sampler_func=losses.binary_sampler, logp_func=losses.bernoulli_loss):
  """Draws a full configuration from a provided latent space coordinate, z, according
to the provided VAE model. Does so based on the provided sampler_func, which should return
a configuration based on the underlying model probability distribution. logp_func should
return the associated log probability under the model. If xSel is provided as a
configuration, will return it and its associated log probability given z. If multiple z
are provided (first dimension is not 1) then one x configuration is returned for each z.
  """
  #Generate the configuration
  vae_output = vae_model.decoder(z)
  if activation is not None:
    vae_output = activation(vae_output)

  if xSel is not None:
    xConf = xSel
  else:
    if isinstance(vae_output, (tuple, list)):
      xConf = sampler_func(*vae_output).numpy()
    else:
      xConf = sampler_func(vae_output).numpy()

  #Calculate log probability of configuration given VAE output parameters...
  #Turns out this is the same as the NEGATIVE reconstruction loss function
  #Maximizing logP, so minimizing -logP as loss
  xlogprob = -logp_func(xConf, vae_output).numpy()

  return xConf, xlogprob


def moveVAE(currConfig, currU, vaeModel, B, energyFunc, zDrawFunc, energyParams={}, samplerParams={}, zDrawType='direct'):
  """Performs a MC move inspired by a VAE model. zDrawFunc should be a class that can be
called with different styles of draws for z.
  """
  #Move in the forward direction
  #Draw a z value from P(z|x1)
  currZ, zLogProbX1 = zDrawLocal(vaeModel, currConfig)
  #Draw a new z value according to chosen distribution
  newZ, z2LogProb = zDrawFunc(draw_type=zDrawType)
  #Now draw new configuration based on new z
  newConfig, logProbX2 = xDrawSimple(vaeModel, newZ, **samplerParams)
  newU = energyFunc(newConfig, **energyParams)

  #Retrace steps in reverse direction
  #Calculate probability of drawing newZ from P(z|x2)
  newZ, zLogProbX2 = zDrawLocal(vaeModel, newConfig, zSel=newZ)
  #Calculate probability of drawing currZ from chosen distribution
  currZ, z1LogProb = zDrawFunc(draw_type=zDrawType, zSel=currZ)
  #Calculate probability of drawing currConfig from currZ
  currConfig, logProbX1 = xDrawSimple(vaeModel, currZ, xSel=currConfig, **samplerParams)

  #Compute log acceptance probability by combining log probabilities
  logPacc = ( -B*(newU - currU)
              + np.nan_to_num(zLogProbX2 - zLogProbX1)
              + np.nan_to_num(z1LogProb - z2LogProb)
              + np.nan_to_num(logProbX1 - logProbX2) )

  #if not np.all(np.isfinite([logprobFor, logprobRev, zLogProbFor, zLogProbRev])):
  print('Breakdown of log(P_acc):')
  print([logPacc, -B*(newU-currU), zLogProbX1, zLogProbX2, z2LogProb, z1LogProb, logProbX2, logProbX1])

  return logPacc, newConfig, newU


class zDrawFunc_wrap_InvNet(object):
  """Based on data, wraps all functions to draw z values. Allows to precompute statistics
given data so that don't have to redo every MC step. Also need to provide a InvNet model for
interpreting the data.
  """
  def __init__(self, dat, inv_model):
    z_model = inv_model(dat)
    self.trueMean = tf.reduce_mean(z_model, axis=0)
    self.trueLogVar = tf.math.log(tf.math.reduce_variance(z_model, axis=0))
    self.minZ = self.trueMean-5*tf.exp(0.5*self.trueLogVar)
    self.maxZ = self.trueMean+5*tf.exp(0.5*self.trueLogVar)

  def __call__(self, draw_type='std_normal', zSel=None, nDraws=1):
    if draw_type == 'std_normal':
      return  zDrawNormal(tf.zeros(self.trueMean.shape),
                          tf.zeros(self.trueLogVar.shape),
                          zSel=zSel, nDraws=nDraws)
    elif draw_type == 'normal':
      return  zDrawNormal(self.trueMean, self.trueLogVar, zSel=zSel, nDraws=nDraws)
    elif draw_type == 'uniform':
      return zDrawUniform(self.minZ, self.maxZ, zSel=zSel, nDraws=nDraws)
    else:
      print('Draw style unknown.')
      return None


def moveInvNet(currConfig, currU, invModel, B, energyFunc, zDrawFunc, energyParams={}, zDrawType='normal'):
  """Performs a MC move inspired by a VAE model. zDrawFunc should be a class that can be
called with different styles of draws for z.
  """
  #Move in the forward direction
  #Draw a z value from the specified distribution
  newZ, zLogProbNew = zDrawFunc(draw_type=zDrawType)
  #And generate new x from this z
  newConfig = invModel(newZ, reverse=True).numpy()
  newU = energyFunc(newConfig, **energyParams)

  #Retrace steps in reverse direction
  #Calculate probability of drawing old z value
  currZ, zLogProbCurr = zDrawFunc(draw_type=zDrawType, zSel=invModel(currConfig).numpy())

  #And that should lead back to the current configuration, so ready to calculate acceptance
  logPacc = ( -B*(newU - currU)
              + np.nan_to_num(zLogProbCurr - zLogProbNew) )
  print('Breakdown of log(P_acc):')
  print(logPacc, -B*(newU-currU), zLogProbNew, zLogProbCurr)

  return logPacc, newConfig, newU


