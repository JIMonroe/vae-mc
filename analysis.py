
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libVAE import dataloaders


def getLatentDists(model, dat, doPlot=False, dataDraws=1, returnSample=False):
  """Determines the distribution of latent variables based on input data.
Assumes a factored Gaussian distribution, so returns means and standard deviation.
  """
  zMeans, zLogvars = model.encoder(dat)
  zSample = np.zeros((dataDraws*dat.shape[0], model.num_latent), dtype='float32')
  for i in range(dataDraws):
    zSample[i*dat.shape[0]:(i+1)*dat.shape[0],:] = model.sampler(zMeans, zLogvars).numpy()

  try:
    zSample, log_det = model.flow(zSample)
    zSample = zSample.numpy()
  except AttributeError:
    pass

  if doPlot:
    distFig, distAx = plt.subplots()
    for i in range(model.num_latent):
      thisHist, thisBins = np.histogram(zSample[:,i], bins='auto', density=True)
      thisCents = 0.5*(thisBins[:-1] + thisBins[1:])
      distAx.plot(thisCents, thisHist, label='%i'%(i+1))
    #And also plot a standard normal distribution
    #This is what the z distribution should be if well trained
    plotZs = np.arange(np.min(zSample), np.max(zSample), 0.001)
    plotVals = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*(plotZs**2))
    distAx.plot(plotZs, plotVals, 'k--', label='Ref')
    distAx.set_xlabel('Latent variable value')
    distAx.set_ylabel('Histogram counts')
    distAx.legend()
    plt.show()

  mean = np.average(zSample, axis=0)
  std = np.std(zSample, ddof=1, axis=0)

  trueMean = tf.reduce_mean(zMeans, axis=0).numpy()
  trueStd = np.sqrt(tf.reduce_mean(tf.math.exp(zLogvars)
                                   + tf.square(zMeans) - tf.square(trueMean), axis=0))
  #Technically NOT a sum of Gaussianly distributed random variables, but instead a
  #sum of Gaussian distributions. So mean is average of means, but std is different.

  print("Sampled mean versus mean mean:")
  print(mean)
  print(trueMean)
  print("Sampled std versus mean std:")
  print(std)
  print(trueStd)

  if returnSample:
    return trueMean, trueStd, zSample
  else:
    #return mean, std
    return trueMean, trueStd


def plotLatent(model, dat, savePlot=False):
  """Plots traversals of all latent dimensions in a model.
  """
  zMean, zStd, zSample = getLatentDists(model, dat, doPlot=True, returnSample=True)
  zPercentiles = np.percentile(zSample, np.linspace(5, 95, 10), axis=0)
  zPercentiles = np.vstack((zPercentiles[:5,:],
                            np.reshape(zMean, (1,-1)),
                            zPercentiles[5:,:]))

  zFig, zAx = plt.subplots(model.num_latent, 11, sharex=True, sharey=True, figsize=(10,10))
  for i in range(model.num_latent):
    zVec = np.reshape(zMean.copy(), (1,-1))
    for j, zVal in enumerate(zPercentiles[:,i]): #Percentiles instead of std
      zVec[0,i] = zVal
      modelOut = tf.nn.sigmoid(model.decoder(tf.cast(zVec, 'float32'))).numpy()
      modelOut = np.squeeze(modelOut)
      randomIm = np.random.random(modelOut.shape)
      imageOut = np.array((modelOut > randomIm), dtype=int)
      zAx[i,j].imshow(imageOut, cmap='gray_r', vmin=0.0, vmax=1.0)
      zAx[i,j].tick_params(axis='both', which='both',
                           left=False, right=False, bottom=False, top=False,
                           labelleft=False, labelbottom=False,
                           labelright=False, labeltop=False)
    zAx[i,0].set_ylabel('%i'%(i+1))
  for j, zVal in enumerate(np.linspace(5, 45, 5)):
    zAx[-1,j].set_xlabel('%2.0f%%'%zVal)
  zAx[-1,5].set_xlabel('mean')
  for j, zVal in enumerate(np.linspace(55, 95, 5)):
    zAx[-1,6+j].set_xlabel('%2.0f%%'%zVal)
  zFig.tight_layout()
  zFig.subplots_adjust(wspace=0.0)
  if savePlot:
    zFig.savefig('latent_traversals.png')
  plt.show()


def plotRecons(model, dat, savePlot=False):
  """Plots reconstructions of provided images.
  """
  fig, ax = plt.subplots(len(list(dat)), 2)
  for i, im in enumerate(dat):
    ax[i,0].imshow(im[:,:,0], cmap='gray_r', vmin=0.0, vmax=1.0)
    thisrecon = model(tf.cast(tf.reshape(im, (1,)+im.shape), 'float32'))
    thisrecon = tf.nn.sigmoid(thisrecon).numpy()
    thisrecon = np.squeeze(thisrecon)
    randomIm = np.random.random(thisrecon.shape)
    thisrecon = np.array((thisrecon > randomIm), dtype=int)
    ax[i,1].imshow(thisrecon, cmap='gray_r', vmin=0.0, vmax=1.0)
    ax[i,0].tick_params(axis='both', which='both',
                        left=False, right=False, bottom=False, top=False,
                        labelleft=False, labelbottom=False,
                        labelright=False, labeltop=False)
    ax[i,1].tick_params(axis='both', which='both',
                        left=False, right=False, bottom=False, top=False,
                        labelleft=False, labelbottom=False,
                        labelright=False, labeltop=False)

  fig.tight_layout()
  if savePlot:
    fig.savefig('reconstructions.png')
  plt.show()


def genConfigData(model, dat, transformFunc=None, nConfs=1000000):
  """Generates configurations according to the VAE model. If transformFunc is provided,
     it applies the function to each generated configuration. Useful for calculating
     appropriately averaged properties by sampling a VAE model.
  """
  #Get latent distribution information and sample in latent space
  zMean, zStd = getLatentDists(model, dat, doPlot=True)
  zMean = tf.cast(zMean, 'float32')
  zStd = tf.cast(zStd, 'float32')
  zSamples = model.sampler(tf.tile(tf.reshape(zMean, (1,)+zMean.shape), (nConfs,1)),
                           tf.tile(tf.reshape(zStd, (1,)+zStd.shape), (nConfs,1)))

  #Generate actual configurations and apply transform
  #Even though it's slow, do one at a time to avoid memory issues
  outDat = []
  for z in zSamples:
    confProbs = tf.math.sigmoid(model.decoder(tf.reshape(z, (1,)+z.shape))).numpy()
    randProbs = np.random.random(size=confProbs.shape)
    conf = np.array((confProbs > randProbs), dtype='int8')
    if transformFunc is not None:
      outDat.append(transformFunc(conf))
    else:
      outData.append(conf)
  outDat = np.array(outDat)

  return outDat


def genReconConfigData(model, dat, transformFunc=None):
  """Generates reconstructions according to the VAE model. If transformFunc is provided,
     it applies the function to each generated configuration. Useful for calculating
     appropriately averaged properties by sampling a VAE model.
  """
  #To avoid memory issues, must do each configuration one at a time
  #For each set of probabilities, generate actual configuration
  outDat = []
  for conf in dat:
    reconProbs = tf.math.sigmoid(model(np.reshape(conf, (1,)+conf.shape))).numpy()
    reconProbs = np.squeeze(reconProbs)
    randProbs = np.random.random(size=reconProbs.shape)
    recon = np.array((reconProbs > randProbs), dtype='int8')
    if transformFunc is not None:
      outDat.append(transformFunc(recon))
    else:
      outData.append(recon)
  outDat = np.array(outDat)

  return outDat


