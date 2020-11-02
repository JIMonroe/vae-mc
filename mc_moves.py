
"""Defines Monte Carlo moves for VAEs and invertible networks.
Useful for testing MC efficiency without actually running MC simulations.
Keeping moves general so will work for any system (hopefully).
"""

import copy
import numpy as np
import tensorflow as tf
from libVAE import vae, dataloaders, losses


def zDrawNormal(mean, logvar, zSel=None, nDraws=1, batch_size=1):
  """Directly draws z from a normal distribution based on mean and logvar
  """
  if zSel is not None:
    if nDraws == 1:
      zval = zSel
    else:
      zval = tf.random.normal(((nDraws-1)*batch_size, mean.shape[0]),
                              mean=mean, stddev=tf.math.exp(0.5*logvar))
      zval = tf.concat((zSel, zval), 0)
  else:
      zval = tf.random.normal((nDraws*batch_size, mean.shape[0]),
                              mean=mean, stddev=tf.math.exp(0.5*logvar))
  zlogprob = tf.reduce_sum( -0.5*tf.math.square(zval - mean)/tf.math.exp(logvar)
                            - 0.5*np.log(2.0*np.pi) - 0.5*logvar,  axis=1).numpy()
  return zval, zlogprob


def zDrawUniform(minZ, maxZ, zSel=None, nDraws=1, batch_size=1):
  """Directly draws from a uniform distribution based on given min and max
  """
  if zSel is not None:
    if nDraws == 1:
      zval = zSel
    else:
      zval = tf.random.uniform(((nDraws-1)*batch_size, minZ.shape[0]),
                               minval=minZ, maxval=maxZ)
      zval = tf.concat((zSel, zval), 0)
  else:
    zval = tf.random.uniform((nDraws*batch_size, minZ.shape[0]),
                             minval=minZ, maxval=maxZ)
  zlogprob = tf.math.reduce_sum(-tf.math.log(maxZ - minZ)).numpy()
  zlogprob = zlogprob*tf.ones(nDraws).numpy()
  return zval, zlogprob


def zDrawDirect(dat_means, dat_logvars, zSel=None, nDraws=1, batch_size=1):
  """Draws a z value from a sample of z means and log variances output by the encoder.
Of course assuming that the sampling is from a normal distribution.
  """
  if zSel is not None:
    if nDraws == 1:
      zval = zSel
    else:
      randInd = np.random.choice(dat_means.shape[0], size=(nDraws-1)*batch_size)
      randMean = tf.gather(dat_means, randInd)
      randLogVar = tf.gather(dat_logvars, randInd)
      zval = tf.random.normal(randMean.shape,
                              mean=randMean, stddev=tf.math.exp(0.5*randLogVar))
      zval = tf.concat((zSel, zval), 0)
  else:
    randInd = np.random.choice(dat_means.shape[0], size=nDraws*batch_size)
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
    self.prior_flow = ('prior' in vae_model.name)

  def __call__(self, draw_type='std_normal', zSel=None, nDraws=1, batch_size=1):
    #First handle batch size - force to match zSel if have zSel
    if zSel is not None:
      batch_size = zSel.shape[0]

    #If have flow and zSel...
    if (self.flow is not None) and (zSel is not None):
      #If draw type is anything other than 'direct' and have prior flow, transform
      if (self.prior_flow) and (draw_type != 'direct'):
        zSel, sel_log_det_rev = self.flow(zSel, reverse=True)
      #If draw type is 'direct' and not prior flow, transform
      elif (not self.prior_flow) and (draw_type == 'direct'):
        zSel, sel_log_det_rev = self.flow(zSel, reverse=True)

    #Now select draw type and draw
    if draw_type == 'std_normal':
      z_draw, log_prob =  zDrawNormal(tf.zeros(self.trueMean.shape),
                                      tf.zeros(self.trueLogVar.shape),
                                      zSel=zSel, nDraws=nDraws, batch_size=batch_size)
    elif draw_type == 'normal':
      z_draw, log_prob = zDrawNormal(self.trueMean, self.trueLogVar,
                                     zSel=zSel, nDraws=nDraws, batch_size=batch_size)
    elif draw_type == 'uniform':
      z_draw, log_prob = zDrawUniform(self.minZ, self.maxZ,
                                      zSel=zSel, nDraws=nDraws, batch_size=batch_size)
    elif draw_type == 'direct':
      z_draw, log_prob = zDrawDirect(self.zMeans, self.zLogvars,
                                     zSel=zSel, nDraws=nDraws, batch_size=batch_size)
    else:
      print('Draw style unknown.')
      return None

    #Now transform back if needed using flow
    if self.flow is not None:
      #If draw type is anything other than 'direct' and have prior flow, transform
      if (self.prior_flow) and (draw_type != 'direct'):
        z_draw, log_det = self.flow(z_draw)
        log_prob -= log_det.numpy()
      #If draw type is 'direct' and not prior flow, transform
      elif (not self.prior_flow) and (draw_type == 'direct'):
        z_draw, log_det = self.flow(z_draw)
        log_prob -= log_det.numpy()

    #If have prior flow and drawing directly, can obtain log probability exactly
    #(don't need to estimate, even though already have in zDrawDirect function)
    if (self.prior_flow) and (draw_type == 'direct'):
      z_prior, log_det = self.flow(z_draw, reverse=True)
      prior_log_prob = -0.5*tf.reduce_sum(tf.square(z_prior)
                                          + tf.math.log(2.0*np.pi),
                                          axis=1).numpy()
      prior_log_prob += log_det.numpy()
      print("Estimated log(P) vs exact with prior flow: %f, %f"%(log_prob, prior_log_prob))
      log_prob = prior_log_prob

    #Given a batch size and draw size, reshape to reflect this
    z_draw = tf.reshape(z_draw, (nDraws, batch_size, z_draw.shape[-1]))
    log_prob = np.reshape(log_prob, (nDraws, batch_size))

    return z_draw, log_prob


def zDrawLocal(vae_model, x, zSel=None, nDraws=1):
  """Draws latent space coordinates given real-space coordinates. This constitutes
a local move, increasing the acceptance probability. Can instead select a z value
and calculate its probability given x.
  """
  #Define batch size
  batch_size = x.shape[0]
  #Generate the mean and log variance of the normal distribution for z given x
  zMean, zLogvar = vae_model.encoder(tf.cast(x, 'float32'))
  if zSel is not None:
    #Must have batch size same as input x
    if zSel.shape[0] != batch_size:
      raise ValueError('Batch size of x is %i and zSel is %i - must match!'
                        %(batch_size, zSel.shape[0]))
    try:
      if 'prior' not in vae_model.name:
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
    if 'prior' not in vae_model.name:
      zval, log_det = vae_model.flow(zval)
      zlogprob -= log_det.numpy()
  except AttributeError:
    pass
  zval = tf.reshape(zval, (nDraws,)+zMean.shape)
  zlogprob = np.reshape(zlogprob, (nDraws, batch_size))
  return zval, zlogprob


def xDrawSimple(vae_model, z, xSel=None, activation=None,
                sampler_func=losses.binary_sampler, logp_func=losses.bernoulli_loss):
  """Draws a full configuration from a provided latent space coordinate, z, according
to the provided VAE model. Does so based on the provided sampler_func, which should return
a configuration based on the underlying model probability distribution. logp_func should
return the associated log probability under the model. If xSel is provided as a
configuration, will return it and its associated log probability given z. If multiple z
are provided (first dimension is not 1) then one x configuration is returned for each z.
  """
  #Generate the configuration
  if xSel is not None and vae_model.autoregress:
    vae_output = vae_model.decoder(z, train_data=tf.cast(xSel, 'float32'))
  else:
    vae_output = vae_model.decoder(z)

  if activation is not None:
    vae_output = activation(vae_output)

  if xSel is not None:
    xConf = tf.cast(xSel, 'float32')
  elif vae_model.autoregress:
    xConf = vae_output[-1]
    if len(vae_output) > 2: #Will be gaussian model, then
      vae_output = vae_output[:-1]
    else:
      vae_output = vae_output[0]
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


def moveVAE(currConfig, currU, vaeModel, B, energyFunc, zDrawFunc, energyParams={}, samplerParams={}, zDrawType='direct', verbose=False):
  """Performs a MC move inspired by a VAE model. zDrawFunc should be a class that can be
called with different styles of draws for z.
  """
  #Move in the forward direction
  #Draw a z value from P(z|x1)
  currZ, zLogProbX1 = zDrawLocal(vaeModel, currConfig, nDraws=1)
  #Since only drawing one sample, NOT config bias, get rid of extra dimension
  currZ = tf.squeeze(currZ, axis=0)
  zLogProbX1 = np.squeeze(zLogProbX1, axis=0)
  #Draw a new z value according to chosen distribution
  newZ, z2LogProb = zDrawFunc(draw_type=zDrawType, nDraws=1, batch_size=currConfig.shape[0])
  newZ = tf.squeeze(newZ, axis=0)
  z2LogProb = np.squeeze(z2LogProb, axis=0)
  #Now draw new configuration based on new z
  newConfig, logProbX2 = xDrawSimple(vaeModel, newZ, **samplerParams)
  newU = energyFunc(newConfig, **energyParams)

  #Retrace steps in reverse direction
  #Calculate probability of drawing newZ from P(z|x2)
  newZ, zLogProbX2 = zDrawLocal(vaeModel, newConfig, zSel=newZ, nDraws=1)
  newZ = tf.squeeze(newZ, axis=0)
  zLogProbX2 = np.squeeze(zLogProbX2, axis=0)
  #Calculate probability of drawing currZ from chosen distribution
  currZ, z1LogProb = zDrawFunc(draw_type=zDrawType, zSel=currZ, nDraws=1)
  currZ = tf.squeeze(currZ, axis=0)
  z1LogProb = np.squeeze(z1LogProb, axis=0)
  #Calculate probability of drawing currConfig from currZ
  currConfig, logProbX1 = xDrawSimple(vaeModel, currZ, xSel=currConfig, **samplerParams)

  #Compute log acceptance probability by combining log probabilities
  logPacc = ( -B*(newU - currU)
              + np.nan_to_num(zLogProbX2 - zLogProbX1)
              + np.nan_to_num(z1LogProb - z2LogProb)
              + np.nan_to_num(logProbX1 - logProbX2) )

  #if not np.all(np.isfinite([logprobFor, logprobRev, zLogProbFor, zLogProbRev])):
  if verbose:
    #print('Breakdown of log(P_acc):')
    full_info = [logPacc, -B*(newU-currU), zLogProbX1, zLogProbX2, z2LogProb, z1LogProb, logProbX2, logProbX1]
    #print(full_info)
    return logPacc, newConfig, newU, full_info
  else:
    return logPacc, newConfig, newU


def vaeBias(vae_model, x, nSample=1000):
  """Biasing function in full-space coordinates to ensure flat sampling along latent-space.
  This requires averaging P(z) over P(z|x) and inverting to obtain the bias. Will return
  the log of the average P(z), so the negated free energy along z associated with x. This bias
  should be added to log probabilities, or -beta*dU. Note that this is a MC estimate of
  the true bias, so will be correct for infinite nSample or over many MC steps. In other
  words, if applied in an MC simulation, will satisfy detailed balance on average, but not
  on every step.
  """
  #Define batch size
  batch_size = x.shape[0]
  #Generate the mean and log variance of the normal distribution for z given x
  zMean, zLogvar = vae_model.encoder(tf.cast(x, 'float32'))
  #And sample nSample times for each x
  z = vae_model.sampler(tf.tile(zMean, (nSample,1)), tf.tile(zLogvar, (nSample,1)))
  log_det = 0.0
  #If have prior flow, need to apply to calculate P(z) in terms of known standard normal
  if 'prior' in vae_model.name:
    z, log_det = vae_model.flow(z, reverse=True)
    log_det = tf.reshape(log_det, (nSample, batch_size, 1))
  z = tf.reshape(z, (nSample,)+zMean.shape)
  #P(z) should be standard normal (except for prior flow, where augment with log_det)
  #Note that bias will be off if flow is poor or model not trained well enough that P(z) is
  #approximately standard normal
  #For numerical stability, subract maximum term
  ln_p_z = -0.5*(z*z) + log_det # - 0.5*np.log(2.0*np.pi)
  max_term = tf.reduce_max(z, axis=0)
  p_z = np.exp(ln_p_z - max_term)
  #Compute bias 
  bias = -(tf.math.log(tf.reduce_mean(p_z, axis=0)) + max_term)
  #Sum over z dimensions (independent Gaussian distributions)
  bias = tf.reduce_sum(bias, axis=1)
  return bias


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


