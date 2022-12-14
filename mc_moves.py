# Written by Jacob I. Monroe, NIST Employee

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
  zlogprob = tf.reduce_sum(-tf.math.log(maxZ - minZ))
  zlogprob = (zlogprob*tf.ones(nDraws*batch_size)).numpy()
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
    #Try to use encoder to get means and log-variances
    try:
      self.zMeans, self.zLogvars = vae_model.encoder(dat)
    #If it doesn't work, encoder is deterministic, so set fixed value for log-variances
    except ValueError:
      self.zMeans = vae_model.encoder(dat)
      #Set value close to machine precision for floating point
      self.zLogvars = tf.fill(self.zMeans.shape, -85.0)
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
    self.prior_flow = (('prior' in vae_model.name) or (vae_model.name == 'cgmodel') or ('fullflow' in vae_model.name))

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
    elif draw_type == 'std_uniform':
      z_draw, log_prob = zDrawUniform(-5.0*tf.ones(self.trueMean.shape),
                                      5.0*tf.ones(self.trueMean.shape),
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
    #if (self.prior_flow) and (draw_type == 'direct'):
    #  z_prior, log_det = self.flow(z_draw, reverse=True)
    #  prior_log_prob = -0.5*tf.reduce_sum(tf.square(z_prior)
    #                                      + tf.math.log(2.0*np.pi),
    #                                      axis=1).numpy()
    #  prior_log_prob += log_det.numpy()
    #  print("Estimated log(P) vs exact with prior flow: %f, %f"%(log_prob, prior_log_prob))
    #  log_prob = prior_log_prob

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
  try:
    zMean, zLogvar = vae_model.encoder(tf.cast(x, 'float32'))
  #If fails, using deterministic encoder, so set log-vars to value close to machine precision
  except ValueError:
    zMean = vae_model.encoder(tf.cast(x, 'float32'))
    zLogvar = tf.fill(zMean.shape, -85.0)
  #Tile so can work with means and logvars more easily throughout
  zMean = tf.tile(zMean, (nDraws, 1))
  zLogvar = tf.tile(zLogvar, (nDraws, 1))
  if zSel is not None:
    #Must have batch size same as input x
    if zSel.shape[0] != batch_size:
      raise ValueError('Batch size of x is %i and zSel is %i - must match!'
                        %(batch_size, zSel.shape[0]))
    try:
      if (('prior' not in vae_model.name) and (vae_model.name != 'cgmodel') and ('fullflow' not in vae_model.name)):
        zSel, sel_log_det_rev = vae_model.flow(zSel, reverse=True)
    except AttributeError:
      pass
    if nDraws == 1:
      zval = zSel
    else:
      #Could use sampler, but no point because only coded for Gaussian
      #Would be less code, but instead be more clear here
      #Also makes code work with deterministic encodings
      # zval = vae_model.sampler(tf.tile(zMean, (nDraws-1,1)), tf.tile(zLogvar, (nDraws-1,1)))
      zval = tf.random.normal((batch_size*(nDraws-1), zMean.shape[-1]),
                              mean=zMean[batch_size:, ...],
                              stddev=tf.exp(0.5*zLogvar[batch_size:, ...]))
      zval = tf.concat((zSel, zval), 0)
  else:
    # zval = vae_model.sampler(tf.tile(zMean, (nDraws,1)), tf.tile(zLogvar, (nDraws,1)))
    zval = tf.random.normal((batch_size*nDraws, zMean.shape[-1]),
                            mean=zMean,
                            stddev=tf.exp(0.5*zLogvar))
  zlogprob = tf.reduce_sum( -0.5*tf.math.square(zval - zMean)/tf.math.exp(zLogvar)
                            - 0.5*np.log(2.0*np.pi) - 0.5*zLogvar, axis=1).numpy()
  try:
    if (('prior' not in vae_model.name) and (vae_model.name != 'cgmodel') and ('fullflow' not in vae_model.name)):
      zval, log_det = vae_model.flow(zval)
      zlogprob -= log_det.numpy()
  except AttributeError:
    pass
  zval = tf.reshape(zval, (nDraws, batch_size, zMean.shape[-1]))
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
      xConf = sampler_func(*vae_output)
    else:
      xConf = sampler_func(vae_output)

  #Calculate log probability of configuration given VAE output parameters...
  #Turns out this is the same as the NEGATIVE reconstruction loss function
  #Maximizing logP, so minimizing -logP as loss
  xlogprob = -logp_func(xConf, vae_output).numpy()

  return xConf.numpy(), xlogprob


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
  if tf.is_tensor(newU):
    newU = newU.numpy()

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


def vaeBias(vae_model, x, nSample=200):
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
  if (('prior' in vae_model.name) or ('fullflow' in vae_model.name)):
    z, log_det = vae_model.flow(z, reverse=True)
    log_det = tf.reshape(log_det, (nSample, batch_size, 1))
  z = tf.reshape(z, (nSample,)+zMean.shape)
  #P(z) should be standard normal (except for prior flow, where augment with log_det)
  #Note that bias will be off if flow is poor or model not trained well enough that P(z) is
  #approximately standard normal
  #For numerical stability, subract maximum term
  ln_p_z = -0.5*(z*z) + log_det # - 0.5*np.log(2.0*np.pi)
  max_term = tf.reduce_max(ln_p_z, axis=0)
  p_z = np.exp(ln_p_z - max_term)
  #Compute bias 
  bias = -(tf.math.log(tf.reduce_mean(p_z, axis=0)) + max_term)
  #Sum over z dimensions (independent Gaussian distributions)
  bias = tf.reduce_sum(bias, axis=1)
  return bias


def moveVAEbiased(currConfig, currU, vaeModel, B, energyFunc, zDrawFunc, energyParams={}, samplerParams={}, zDrawType='uniform', verbose=False):
  """Augments moveVAE with bias along latent space coordinate based on VAE model. May be best
to use uniform z-sampling with this scheme, but will need to test.
  """
  #First call regular VAE move, then compute bias and augment log probability
  move_info = moveVAE(currConfig, currU, vaeModel, B, energyFunc, zDrawFunc,
                      energyParams=energyParams, samplerParams=samplerParams,
                      zDrawType=zDrawType, verbose=verbose)
  logPacc = move_info[0]
  newConfig = move_info[1]
  newU = move_info[2]
  bias_curr = vaeBias(vaeModel, currConfig).numpy()
  bias_new = vaeBias(vaeModel, newConfig).numpy()
  logPacc = logPacc + (bias_new - bias_curr)
  if verbose:
    full_info = move_info[3]
    full_info += [bias_curr, bias_new]
    return logPacc, newConfig, newU, full_info
  else:
    return logPacc, newConfig, newU


def moveVAE_cb(currConfig, currU, vaeModel, B, energyFunc, zDrawFunc, energyParams={}, samplerParams={}, zDrawType='direct', n_draws=100, verbose=False):
  """Performs a MC move inspired by a VAE model with configurational bias based on treating
the full model P(x) as the arbitrary trial distribution. zDrawFunc should be a class that can
be called with different styles of draws for z.
  """
  n_batch = currConfig.shape[0]

  #Move in the forward direction
  #Draw a z value from P(z|x1)
  #Do for just a single z for now, since other x generated in reverse will also have z
  currZ, zLogProbX1 = zDrawLocal(vaeModel, currConfig, nDraws=1)
  currZ = tf.squeeze(currZ, axis=0)
  zLogProbX1 = np.squeeze(zLogProbX1, axis=0)
  #Draw a new z value according to chosen distribution
  newZ, z2LogProb = zDrawFunc(draw_type=zDrawType, nDraws=n_draws,
                              batch_size=n_batch)
  newZ = tf.reshape(newZ, (-1, newZ.shape[-1]))
  #Now draw new configuration based on new z, note drawing over all n_draws and batches
  newConfig, logProbX2 = xDrawSimple(vaeModel, newZ, **samplerParams)
  newU = energyFunc(newConfig, **energyParams)
  if tf.is_tensor(newU):
    newU = newU.numpy()

  #Retrace steps in reverse direction
  #Calculate probability of drawing newZ from P(z|x2)
  #Exactly retracing steps, so nDraws is 1 because not drawing extra
  #Still shaped so n_draws and batch_size compressed on same axis
  newZ, zLogProbX2 = zDrawLocal(vaeModel, newConfig, zSel=newZ, nDraws=1)
  #Already have single z value, but want to generate more in reverse
  #Calculate probability of drawing currZ from chosen distribution
  currZ, z1LogProb = zDrawFunc(draw_type=zDrawType, zSel=currZ, nDraws=n_draws)
  #Calculate probability of drawing currConfig from currZ
  #Need to handle differently for actual current config and proposed reverse configs
  #Current should be first in currZ, since it was zSel, then others are after
  currConfig, logProbX1 = xDrawSimple(vaeModel, currZ[0, ...],
                                      xSel=currConfig, **samplerParams)
  #No selection here, just draw
  revConfig, revlogProbX1 = xDrawSimple(vaeModel,
                                        tf.reshape(currZ[1:, ...], (-1, currZ.shape[-1])),
                                        **samplerParams)
  #Need to calculate energy for proposed reverse configs
  revU = energyFunc(revConfig, **energyParams)
  if tf.is_tensor(revU):
    revU = revU.numpy()
  #And finally need to compute P(z|x) for all newly generated reverse configurations
  revCurrZ, revzLogProbX1 = zDrawLocal(vaeModel, revConfig,
                                       zSel=tf.reshape(currZ[1:, ...], (-1, currZ.shape[-1])),
                                       nDraws=1)

  #Now reshape all energies and log probabilities so n_draws is on first axis
  #And will put together reverse energies and log probs
  revU = np.reshape(revU, (n_draws-1, n_batch))
  revU = np.concatenate([currU[np.newaxis, :], revU])
  revlogProbX1 = np.reshape(revlogProbX1, (n_draws-1, n_batch))
  revlogProbX1 = np.concatenate([logProbX1[np.newaxis, ...], revlogProbX1])
  revzLogProbX1 = np.reshape(revzLogProbX1, (n_draws-1, n_batch))
  revzLogProbX1 = np.concatenate([zLogProbX1[np.newaxis, ...], revzLogProbX1])
  newU = np.reshape(newU, (n_draws, n_batch))
  logProbX2 = np.reshape(logProbX2, (n_draws, n_batch))
  zLogProbX2 = np.reshape(zLogProbX2, (n_draws, n_batch))
  z1LogProb = np.reshape(z1LogProb, (n_draws, n_batch))
  newConfig = np.reshape(newConfig, (n_draws,)+currConfig.shape)

  #For forward direction, select configuration based on config bias weights
  log_cb_for = -B*newU - logProbX2 - z2LogProb + zLogProbX2
  #Use Gumbel max trick
  sel_ind = np.argmax(log_cb_for + np.random.gumbel(size=log_cb_for.shape), axis=0)
  newConfig = np.squeeze(np.take_along_axis(newConfig,
                                            sel_ind[np.newaxis, :, np.newaxis], axis=0),
                         axis=0)
  newU = np.squeeze(np.take_along_axis(newU,
                                       sel_ind[np.newaxis, :], axis=0),
                    axis=0)
  #And sum config bias log weights to get Rosenbluth weight
  max_log_cb_for = np.max(log_cb_for, axis=0)
  w_for = max_log_cb_for + np.sum(np.exp(log_cb_for - max_log_cb_for), axis=0)

  #In reverse, just need to sum to get Rosenbluth weight, W
  log_cb_rev = -B*revU - revlogProbX1 - z1LogProb + revzLogProbX1
  max_log_cb_rev = np.max(log_cb_rev, axis=0)
  w_rev = max_log_cb_rev + np.sum(np.exp(log_cb_rev - max_log_cb_rev), axis=0)

  #Compute log acceptance probability by combining log probabilities
  logPacc = np.nan_to_num(w_for) - np.nan_to_num(w_rev)

  #if not np.all(np.isfinite([logprobFor, logprobRev, zLogProbFor, zLogProbRev])):
  if verbose:
    #print('Breakdown of log(P_acc):')
    full_info = [logPacc, w_rev, w_for]
    #print(full_info)
    return logPacc, newConfig, newU, full_info
  else:
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


def moveGauss(currConfig, currU, B, energyFunc, energyParams={}, noise=0.02):
  """General purpose MC move that simply adds normally-distributed noise to an input
configuration. The amount of noise can be modified by setting the noise input.
  """
  norm_sample = noise*np.random.normal(size=currConfig.shape).astype(currConfig.dtype)
  newConfig = currConfig + norm_sample
  newU = energyFunc(newConfig, **energyParams)
  if tf.is_tensor(newU):
    newU = newU.numpy()
  logPacc = -B*(newU - currU)
  return logPacc, newConfig, newU


