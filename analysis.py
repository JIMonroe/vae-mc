
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libVAE import dataloaders

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


def getLatentDists(model, dat, doPlot=False, dataDraws=1, returnSample=False):
  """Determines the distribution of latent variables based on input data.
Assumes a factored Gaussian distribution, so returns means and standard deviation.
  """
  zSample = np.zeros((dataDraws*dat.shape[0], model.num_latent), dtype='float32')
  try:
    zMeans, zLogvars = model.encoder(dat)
    for i in range(dataDraws):
      zSample[i*dat.shape[0]:(i+1)*dat.shape[0],:] = model.sampler(zMeans, zLogvars).numpy()
  except ValueError:
    #If using deterministic mapping or adversarial VAE, will end up here
    for i in range(dataDraws):
      zSample[i*dat.shape[0]:(i+1)*dat.shape[0],:] = model.encoder(dat).numpy()

  try:
    if model.name not in ['fullflow_vae', 'priorflow_vae', 'cgmodel', 'dimercg']:
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

  try:
    trueMean = tf.reduce_mean(zMeans, axis=0).numpy()
    trueStd = np.sqrt(tf.reduce_mean(tf.math.exp(zLogvars)
                                     + tf.square(zMeans) - tf.square(trueMean), axis=0))
    #Technically NOT a sum of Gaussianly distributed random variables, but instead a
    #sum of Gaussian distributions. So mean is average of means, but std is different.
  except NameError:
    trueMean = mean
    trueStd = std

  print("Sampled mean versus mean mean:")
  print(mean)
  print(trueMean)
  print("Sampled std versus mean std:")
  print(std)
  print(trueStd)

  if returnSample:
    return trueMean, trueStd, zSample
  else:
    return trueMean, trueStd


def plotLatent(model, dat, savePlot=False):
  """Plots traversals of all latent dimensions in a model.
  """
  zMean, zStd, zSample = getLatentDists(model, dat, doPlot=False, returnSample=True)
  percentVals = [5, 25, 50, 75, 95]
  zPercentiles = np.percentile(zSample, percentVals, axis=0)
  #zPercentiles = np.vstack((zPercentiles[:5,:],
  #                          np.reshape(zMean, (1,-1)),
  #                          zPercentiles[5:,:]))
  zMedian = zPercentiles[2, :]

  zFig, zAx = plt.subplots(model.num_latent, zPercentiles.shape[0],
                           sharex=True, sharey=True,
                           figsize=(0.95*zPercentiles.shape[0], 1.05*model.num_latent),
                           dpi=300)
  if  len(zAx.shape) == 1:
    zAx = np.array([zAx])
  for i in range(model.num_latent):
    #zVec = np.reshape(zMean.copy(), (1,-1))
    zVec = np.reshape(zMedian.copy(), (1,-1))
    for j, zVal in enumerate(zPercentiles[:,i]): #Percentiles instead of std
      zVec[0,i] = zVal
      modelOut = model.decoder(tf.cast(zVec, 'float32'))
      if isinstance(modelOut, tuple):
        modelOut = modelOut[1]
      else:
        modelOut = tf.nn.sigmoid(modelOut).numpy()
      modelOut = np.squeeze(modelOut)
      randomIm = np.random.random(modelOut.shape)
      imageOut = np.array((modelOut > randomIm), dtype=int)
      zAx[i,j].imshow(imageOut, cmap='gray_r', vmin=0.0, vmax=1.0)
      zAx[i,j].tick_params(axis='both', which='both',
                           left=False, right=False, bottom=False, top=False,
                           labelleft=False, labelbottom=False,
                           labelright=False, labeltop=False)
    zAx[i,0].set_ylabel('%i'%(i+1))
  for j, pVal in enumerate(percentVals):
    zAx[-1,j].set_xlabel('%2.0f%%'%pVal)
  #for j, zVal in enumerate(np.linspace(5, 45, 5)):
  #  zAx[-1,j].set_xlabel('%2.0f%%'%zVal)
  #zAx[-1,5].set_xlabel('mean')
  #for j, zVal in enumerate(np.linspace(55, 95, 5)):
  #  zAx[-1,6+j].set_xlabel('%2.0f%%'%zVal)
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
    if isinstance(thisrecon, tuple):
      #If returns list, either not lattice gas, or autoregressive (so already 1's and 0's)
      thisrecon = thisrecon[1]
    else:
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


#Some functions specific to lattice gas system
def alignConfsBrute(conf0, conf1):
  """Given two configurations, aligns them based on squared error.
Only considers 90 degree rotations or ineteger translations, so uses brute force.
For a 64x64 array this means generating 4x64x64 arrays and computing squared error
for each with conf0. The minimum is then selected. Can provide multiple configurations
in conf1, but be careful with MEMORY.
  """

  rot0 = conf1
  rot1 = np.rot90(conf1, axes=(1,2))
  rot2 = np.rot90(rot1, axes=(1,2))
  rot3 = np.rot90(rot2, axes=(1,2))
  allRots = np.array([rot0, rot1, rot2, rot3])

  minConf = np.zeros(conf1.shape)
  minSE = np.ones(conf1.shape[0])*np.finfo('float').max
  for i in range(conf0.shape[0]):
    for j in range(conf0.shape[1]):
      confs = np.roll(allRots, (i,j), axis=(2,3))
      thisSE = np.sum((confs - conf0)**2, axis=(2,3))
      minInds = np.argmin(thisSE, axis=0)
      replaceInds = np.where(thisSE[minInds, np.arange(thisSE.shape[1])] < minSE)[0]
      minConf[replaceInds] = confs[minInds[replaceInds], replaceInds, :, :]
      minSE[replaceInds] = thisSE[minInds[replaceInds], replaceInds]

  return minConf


def createConnectivityMatrix(conf):
  """Builds connectivity matrix for sklearn agglomerative clustering.
  """
  #Need to build connectivity graph like in sklearn, but need to include periodicity...
  #connectivity = grid_to_graph(conf.shape[0], conf.shape[1])
  L = conf.shape[0]
  nPix = L*L
  connectivity = np.zeros((nPix, nPix), dtype=int)
  #So if L=64, then...
  #Numbering is row then column, so (1,0) is 64 and (1,63) is 127
  for i in range(connectivity.shape[0]):
    ind0 = i #This pixel
    if i%L == 0: #On left edge
      ind1 = i+(L-1)
    else:
      ind1 = i-1 #1 back in row
    if i%L == (L-1): #On right edge
      ind2 = i-(L-1)
    else:
      ind2 = i+1 #1 forward in row
    if i//L == 0: #On top edge
      ind3 = i+(nPix-L)
    else:
      ind3 = i-L #1 up in column
    if i//L == (L-1): #On bottom edge
      ind4 = i-(nPix-L)
    else:
      ind4 = i+L #1 down in column
    connectivity[i,[ind0, ind1, ind2, ind3, ind4]] = 1

  return connectivity


def getClusterInfo(conf, connectivity=None, showPlot=False):
  """Given an image, uses scikit-learn to identify clusters.
  """

  if connectivity is None:
    connectivity = createConnectivityMatrix(conf)

  #And now do clustering
  clustInfo = AgglomerativeClustering(n_clusters=None,
                                      linkage='single',
                                      connectivity=connectivity,
                                      distance_threshold=1)
  toFit = np.reshape(conf, (-1,1))
  clustInfo.fit(toFit)

  if showPlot:
    lab = np.reshape(clustInfo.labels_, conf.shape)
    fig, ax = plt.subplots()
    ax.imshow(conf, cmap='gray_r')
    for l in range(clustInfo.n_clusters_):
      ax.contour(lab == l, colors=[plt.cm.nipy_spectral(l/float(clustInfo.n_clusters_)), ])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

  #We're interested in the number of clusters and cluster size of each cluster
  #And want to distinguish clusters of particles versus empty lattice sites
  clustSizeUn = []
  clustSizeOc = []
  for l in range(clustInfo.n_clusters_):
    thisMembers = np.where(clustInfo.labels_ == l)[0]
    if toFit[thisMembers[0]] == 0:
      clustSizeUn.append(len(thisMembers))
    else:
      clustSizeOc.append(len(thisMembers))

  return np.array(clustSizeUn), np.array(clustSizeOc)



