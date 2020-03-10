
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libVAE import dataloaders


def getLatentDists(model, dat, doPlot=False):
  """Determines the distribution of latent variables based on input data.
Assumes a factored Gaussian distribution, so returns means and standard deviation.
  """
  #dat = dataloaders.raw_image_data(dataFile)
  zMeans, zLogvars = model.encoder(dat)
  zSample = model.sampler(zMeans, zLogvars).numpy()

  if doPlot:
    distFig, distAx = plt.subplots()
    for i in range(model.num_latent):
      thisHist, thisBins = np.histogram(zSample[:,i], bins='auto')
      thisCents = 0.5*(thisBins[:-1] + thisBins[1:])
      distAx.plot(thisCents, thisHist, label='%i'%(i+1))
    distAx.set_xlabel('Latent variable value')
    distAx.set_ylabel('Histogram counts')
    distAx.legend()
    plt.show()

  mean = np.average(zSample, axis=0)
  std = np.std(zSample, ddof=1, axis=0)

  return mean, std


def plotLatent(model, dat, savePlot=False):
  """Plots traversals of all latent dimensions in a model.
  """
  zMean, zStd = getLatentDists(model, dat, doPlot=True)

  zFig, zAx = plt.subplots(model.num_latent, 10, sharex=True, sharey=True, figsize=(9.5,10))
  for i in range(model.num_latent):
    zVec = np.reshape(zMean, (1,-1))
    for j, zVal in enumerate(np.linspace(-2.0, 2.0, 10)):
      zVec[0,i] = zMean[i] + zVal*zStd[i]
      modelOut = tf.nn.sigmoid(model.decoder(zVec)).numpy()
      modelOut = np.squeeze(modelOut)
      randomIm = np.random.random(modelOut.shape)
      imageOut = np.array((modelOut > randomIm), dtype=int)
      zAx[i,j].imshow(imageOut, cmap='gray_r', vmin=0.0, vmax=1.0)
      zAx[i,j].tick_params(axis='both', which='both',
                           left=False, right=False, bottom=False, top=False,
                           labelleft=False, labelbottom=False,
                           labelright=False, labeltop=False)
    zAx[i,0].set_ylabel('%i'%(i+1))
  for j, zVal in enumerate(np.linspace(-2.0, 2.0, 10)):
    zAx[-1,j].set_xlabel('z=%1.1f'%zVal)
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
    thisrecon = model(tf.cast(tf.reshape(im, (1,64,64,1)), 'float32'))
    thisrecon = tf.nn.sigmoid(thisrecon).numpy()
    thisrecon = np.squeeze(thisrecon)
    randomIm = np.random.random(thisrecon.shape)
    #thisrecon = np.array((thisrecon > randomIm), dtype=int)
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


