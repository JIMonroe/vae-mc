
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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


#Analyses to examine properties of the latent space
#All before and in LatentAnalysis class assume have trained prior flow in VAE
def utilized_latent_dims(z_means, z_logvars):
    """
    Assuming multivariate Gaussian encoding distributions with diagonal covariance matrix (independent
    z conditioned on x), computes the average of the Jeffreys Divergence between all pairs of encoding
    distributions (i.e., all pairs of x samples). This quantity is output separately for each latent
    dimension, which is appropriate for the assumed distributions. For latent dimensions that are not
    utilized, we expect the average JD to be small because on average encoding distributions overlap
    significantly and do not differentiate between x samples. Important latent dimensions, on the other
    hand, are expected to have low variance of the encoding distribution compared to the overall latent
    space along that dimension, leading to high average JD.

    Inputs:
            z_means - means of encoding distributions for each x sample (N_samples x N_latent)
            z_logvars - log-variances of encoding distributions (N_samples x N_latent)
    Ouputs:
            jd - average Jeffreys Divergence between encoding distributions for each dimension (N_latent)
    """

    jd = []

    #Loop over latent dimensions
    for i in range(tf.shape(z_means)[1]):

        #To parallelize computation, transpose
        diff_means = z_means[:, i:i+1] - tf.transpose(z_means[:, i:i+1])
        diff_logvars = z_logvars[:, i:i+1] - tf.transpose(z_logvars[:, i:i+1])
        ratio_vars = tf.math.exp(-diff_logvars)

        #Formula is KL(N_1 | N_2) = 0.5*(var_1/var_2 + (1/var_2)(mean_2 - mean_1)^2 - 1 + log(var_2/var_1))
        #for single dimension
        KL_mat = 0.5*(ratio_vars + (diff_means**2)*tf.math.exp(-z_logvars[:, i:i+1]) - 1.0 + diff_logvars)

        #Above is matrix of KL divergences from i to j, so KL_mat[i, j] = KL(N_j, N_i)
        #(should be zero on the diagonals!)
        #So to get Jeffreys, JD = 0.5*(KL(N_i, N_j) + KL(N_j, N_i))
        JD_mat = 0.5*(KL_mat + tf.transpose(KL_mat))

        #JD_mat should now be symmetric with zero diagonals
        #Want average over all unique pairs of distributions from unique x samples
        #(excluding distributions with themselves)
        #Since diagonal terms contribute zero, just divide by 2 times total number of unique pairs
        #(or N*N - N, all pairs minus diagonal)
        jd.append(tf.reduce_sum(JD_mat)
                  / tf.cast(tf.shape(z_means)[0]**2 - tf.shape(z_means)[0], dtype=JD_mat.dtype))

    if len(jd) == 1:
        jd = [jd]

    return tf.concat(jd, axis=0)


def latent_linear_independence(z_sample):

    """
    Given a sample in the latent space, computes the covariance matrix and performs PCA.

    Inputs:
            z_sample - sample in latent space
    Outputs:
            cov_mat - covariance matrix
            eigvecs - eigenvectors of covariance matrix (so principal components)
            eigvals - eigenvalues of covariance matrix
            var_explained - variance explained by each eigenvalue
            gauss_tot_corr - Gaussian approximation to total correlation of latent space (used by Locatello, et al. 2019)
    """

    #First put dimensions on equal footing in terms of scale and range (get average and std)
    avg_z = tf.reduce_mean(z_sample, axis=0)
    std_z = (tf.math.reduce_std(z_sample, axis=0)
             *tf.cast(tf.shape(z_sample)[0] / (tf.shape(z_sample)[0] - 1), z_sample.dtype)) #Note divides by N, not N-1

    #Wrap numpy for covariance matrix for latent space sample
    cov_mat = tfp.stats.covariance((z_sample - avg_z) / std_z)

    #Could just look at structure of covariance matrix
    #But for more info, look at principal components
    eigvals, eigvecs = tf.linalg.eigh(cov_mat)

    #Explained variance for PCs
    var_explained = eigvals / tf.reduce_sum(eigvals)

    #Finish Gaussian approximation with raw covariance matrix
    raw_cov_mat = tfp.stats.covariance(z_sample)
    #And total correlation for Guassian is sum of individual dimension entropies minus full entropy
    #Which simplifies to just Sum(0.5*ln(sigma_i^2)) - 0.5*ln(det(Sigma))
    gauss_tot_corr = tf.reduce_sum(tf.math.log(std_z)) - 0.5*tf.linalg.logdet(raw_cov_mat)

    return cov_mat, eigvecs, eigvals, var_explained, gauss_tot_corr


def total_corr_TCVAE(z_samples, z_means, z_logvars):

    """
    Computes total correlation as a batch estimate as developed in Chen, et al. 2019 (TCVAE paper) and
    implemented in the disentanglement lib by Locatello, et al. 2019. Note that, though this quantity
    approximates a KL divergence, it is a lower bound and thus may not be positive.

    Inputs:
            z_samples - samples from q(z) via q(z|x) over many x
            z_means - means of q(z|x) from which each z sample were generated
            z_logvars - log-variances from which z samples were generated
    Outputs:
            total_corr - total correlation estimate, which is lower bound on true value
    """
    z_samples = tf.expand_dims(z_samples, 1)
    z_means = tf.expand_dims(z_means, 0)
    z_logvars = tf.expand_dims(z_logvars, 0)
    #Log probabilities of all z samples evaluated in all q(z|x) distributions
    log_qz_prob = -0.5*(tf.square(z_samples - z_means)*tf.math.exp(-z_logvars)
                        + z_logvars
                        + tf.math.log(tf.cast(2.0*np.pi, z_means.dtype)))
    #Product of marginals (or sum of log marginal probabilities)
    log_qz_product = tf.reduce_sum(tf.math.reduce_logsumexp(log_qz_prob, axis=1), axis=1)
    #Estimate of full log probability
    log_qz = tf.math.reduce_logsumexp(tf.reduce_sum(log_qz_prob, axis=2), axis=1)
    return tf.reduce_mean(log_qz - log_qz_product)


def latent_info_content(z_sample, tz_sample, log_det_rev, std_norm_sample, t_std_norm_sample, log_det_std_norm):

    """
    If have trained flow, can compute two estimates of relative entropy, one in each direction.
    If sample in standard normal, P(z'), approximate Srel as

    -S_sn - <ln[sn(finv(z'))*J(z')]>_sn

    where z' is drawn from the standard normal, S_sn is the entropy of the standard normal, finv()
    is the inverse flow transformation, but acts on z', and J(z') is the associated log determinant.
    This assumes q(z) can be well-approximated by P(z')J(z'). Alternatively, can sample empirical
    distribution q(z). In this case, have a number of options, but will take

    <ln[sn(finv(z))]>_q(z) + <lnJ(z)>_q(z) - <ln[sn(z)]>_q(z)

    This assumes that q(z') is approximately P(z'), which is true for a well-trained flow. Note that
    the first average is the log-probability of the standard normal averaged over the transformed
    version of z, and the second is over the untransformed. Computing in two directions may or may not
    be helpful, which is why both KL divergences and the Jeffreys divergence are returned. Even if
    the flow is well-learned, there is no reason for the two KL divergences to be similar since this
    is intrinsically a non-symmetric metric. However, using relative entropy versus entropy is
    preferred - first, we get estimates in two directions and second, KL is bounded by zero, so if
    zero it means no additional info than standard normal. This should be caveated, however, with the
    fact that a meaningful latent space could still exist but be in the shape of standard normal.
    If the latent distribution is more convoluted than standard normal, however, we are definitely
    encoding more information. So more precisely this assesses the non-standard-normal-ness of q(z).
    If have similarity to standard normal AND latent dimensions are all unused, then no info encoded.
    Still returning estimate of entropy of q(z), because compute it to get KL_rev.

    Inputs:
            z_sample - sample from q(z)
            tz_sample - transformed sample from z to z' (so finv(z))
            log_det_rev - log determinant of transform from q(z) to q(z')
            std_norm_sample - sample of z' from standard normal distribution P(z')
            t_std_norm_sample - tranformation of standard normal sample in REVERSE direction, finv(z')
            log_det_std_norm - log determinant of transform from standard normal sample
    Outputs:
            S_q - upper bound estimate of entropy of q(z)
            KL_rev - KL divergence in "reverse" direction, specifically KL(q(z), sn(z))
            KL_for - KL divergence in "forward" direction, so KL(sn(z), q(z))
            JD - Jeffreys divergence 1/2(KLfor + KLrev)
            tot_corr - total correlation KL(q(z), prod(q(zi))), or sum(S_qzi) - S_qz
    """

    #Compute upper bound on entropy of q(z)
    S_q = -tf.reduce_mean(tf.reduce_sum(-0.5*tf.square(tz_sample)
                                        - tf.cast(0.5*tf.math.log(2.0*np.pi), tz_sample.dtype), axis=1)
                          + log_det_rev)
    #Since entropy estimate is upper bound and negate it, KL_rev is lower bound
    KL_rev = -S_q - tf.reduce_mean(tf.reduce_sum(-0.5*tf.square(z_sample)
                                                 - tf.cast(0.5*tf.math.log(2.0*np.pi), z_sample.dtype), axis=1)
                                   ) #Terms involving 2*pi should cancel
    sn_ent = (0.5*tf.cast(tf.shape(std_norm_sample)[1], t_std_norm_sample.dtype)
                 *tf.cast(tf.math.log(2.0*np.pi) + 1.0, t_std_norm_sample.dtype)
             )
    KL_for = (-sn_ent
              - tf.reduce_mean(tf.reduce_sum(-0.5*tf.square(t_std_norm_sample)
                                             - tf.cast(0.5*tf.math.log(2.0*np.pi), t_std_norm_sample.dtype), axis=1)
                               + log_det_std_norm)
             )
    JD = 0.5*(KL_rev + KL_for) #Technically, I think Jeffreys Divergence is just the sum, but 1/2 makes sense
    #Can also compute total correlation if compute dimension-wise entropies
    #Since each is 1D, can do reasonably well with just histogramming
    S_ind = 0.0
    for k in range(z_sample.shape[1]):
        this_hist, this_bins = np.histogram(z_sample[:, k], bins='auto', density=True)
        #Exclude bins with no counts
        non_zero_bins = (this_hist != 0.0)
        S_ind -= np.sum(this_hist[non_zero_bins]*np.log(this_hist[non_zero_bins])
                        *(this_bins[1:] - this_bins[:-1])[non_zero_bins])
    tot_corr = S_ind - S_q
    return S_q, KL_rev, KL_for, JD, tot_corr


class LatentAnalysis(object):
    """
    Wraps all analyses specific to the latent space into one class. Results for
    analyses will be stored in a dictionary.
    """

    def __init__(self, vae_model, x_input):
        """
        Inputs:
                vae_model - vae model to analyze
                x_input - input samples from full-space distribution for vae model
        Outputs:
                LatentAnalysis class object
        """
        self.vae_model = vae_model
        #For all analyses, will need means and log-vars, even if just to sample
        #Compute encoding distributions for all inputs
        #If want to bootstrap, just resample the means and log-vars instead of x
        self.z_means, self.z_logvars = self.vae_model.encoder(x_input)
        self.n_obs = tf.shape(self.z_means)[0]

        #Will need random number generator
        self.rng = tf.random.Generator.from_non_deterministic_state() #self.rng = np.random.default_rng()

        #Set up dictionary for results, all currently None
        self.result_keys = ['qzx_jd_per_dim',
                            'qz_cov_mat',
                            'qz_cov_eigvecs',
                            'qz_cov_eigvals',
                            'qz_var_explained',
                            'qz_gauss_tot_corr',
                            'qz_tot_corr_tcvae',
                            'qz_tot_corr',
                            'kl_rev',
                            'kl_for',
                            'qz_jd',
                            'qz_s',]
        self.results = dict(zip(self.result_keys, [None]*len(self.result_keys)))
        #For uncertainty, nest it as dict within results with same keys
        self.results['uncertainty'] = dict(zip(self.result_keys, [None]*len(self.result_keys)))

    def _all_analyses_(self, indices=None, n_per=1):
        """
        Performs all analyses

        Inputs:
                indices - (optional) indices of data used to generated q(z|x) means
                          and logvars to use for all analysis
                n_per - (1) number of z samples to draw per q(z|x) distribution
        Outputs:
                out_means - Tuple of all analysis results; means if analysis used batching
                out_vars - Tuple of variances for only those analyses with batching
        """
        #Use all of q(z|x) distributions once if not specified
        if indices is None:
            indices = tf.range(self.n_obs)

        this_z_means = tf.gather(self.z_means, indices)
        this_z_logvars = tf.gather(self.z_logvars, indices)

        #Perform all analyses
        #Need samples for some analyses
        #And make sure if using regular AE, don't actually sample
        if self.vae_model.sample_latent:
            z_sample = []
            for k in range(n_per):
                z_sample.append(self.vae_model.sampler(this_z_means, this_z_logvars))
            z_sample = tf.concat(z_sample, axis=0)
            this_z_means = tf.tile(this_z_means, (n_per, 1))
            this_z_logvars = tf.tile(this_z_logvars, (n_per, 1))
        else:
            z_sample = this_z_means + this_z_logvars

        #Do analyses that require batching (to prevent overflows)
        #But skip if standard autoencoder because requires means and logvars of q(z|x)
        if self.vae_model.sample_latent:
            batch_size = 500
            qzx_jd_per_dim = []
            qz_tot_corr_tcvae = []
            num_batch = []
            for i in range(0, tf.shape(this_z_means)[0], batch_size):
                start = i
                end = i+batch_size
                qzx_jd_per_dim.append(utilized_latent_dims(this_z_means[start:end],
                                                           this_z_logvars[start:end]))
                qz_tot_corr_tcvae.append(total_corr_TCVAE(z_sample[start:end],
                                                          this_z_means[start:end],
                                                          this_z_logvars[start:end]))
                num_batch.append(tf.cast(tf.shape(this_z_means[start:end])[0], this_z_means.dtype))
            num_batch = tf.stack(num_batch)
            batch_weights = num_batch / tf.reduce_sum(num_batch)
            qzx_jd_per_dim = tf.stack(qzx_jd_per_dim)
            qz_tot_corr_tcvae = tf.stack(qz_tot_corr_tcvae)
            qzx_jd_per_dim_mean = tf.reduce_sum(qzx_jd_per_dim*tf.reshape(batch_weights, (-1, 1)), axis=0)
            qz_tot_corr_tcvae_mean = tf.reduce_sum(qz_tot_corr_tcvae*batch_weights)
            #Batching accounts for large part of uncertainty in these quantities, so estimate and return
            qzx_jd_per_dim_var = tf.reduce_sum(tf.square(qzx_jd_per_dim - qzx_jd_per_dim_mean)*tf.reshape(batch_weights, (-1, 1)), axis=0)
            qz_tot_corr_tcvae_var = tf.reduce_sum(tf.square(qz_tot_corr_tcvae - qz_tot_corr_tcvae_mean)*batch_weights)
        #Can't compute these metrics with standard AE
        else:
            qzx_jd_per_dim_mean = tf.constant(np.nan)
            qz_tot_corr_tcvae_mean = tf.constant(np.nan)
            qzx_jd_per_dim_var = tf.constant(np.nan)
            qz_tot_corr_tcvae_var = tf.constant(np.nan)
        #print('Checked utilized latent dims')
        #print('Checked total correlation')

        #And those that don't
        linear_ind_info = latent_linear_independence(z_sample)
        qz_cov_mat = linear_ind_info[0]
        qz_cov_eigvecs = linear_ind_info[1]
        qz_cov_eigvals = linear_ind_info[2]
        qz_var_explained = linear_ind_info[3]
        qz_gauss_tot_corr = linear_ind_info[4]
        #print('Checked linear independence')

        #And extra sampling for information content estimates (for KL divergence in both directions from standard normal)
        std_norm_sample = self.rng.normal(z_sample.shape)
        tz_sample, log_det_rev = self.vae_model.flow(z_sample, reverse=True)
        t_std_norm_sample, log_det_std_norm = self.vae_model.flow(std_norm_sample, reverse=True)
        #print('Finished both flows')
        qz_s, kl_rev, kl_for, qz_jd, tot_corr = latent_info_content(z_sample, tz_sample, log_det_rev,
                                                         std_norm_sample, t_std_norm_sample, log_det_std_norm)
        #print('Estimated information content with flows')

        out_means = (qzx_jd_per_dim_mean,
                     qz_cov_mat,
                     qz_cov_eigvecs, qz_cov_eigvals, qz_var_explained,
                     qz_gauss_tot_corr, qz_tot_corr_tcvae_mean, tot_corr,
                     kl_rev, kl_for, qz_jd, qz_s)
        out_vars = (qzx_jd_per_dim_var, qz_tot_corr_tcvae_var)
        return out_means, out_vars

    def _bootstrap_(self, n_boot=100):
        """
        Function to bootstrap the analysis, re-running _all_analyses_ for each bootstrap.

        Inputs:
                n_boot - (100) number of bootstrap samples to generate and analyze
        Outputs:
                boot_avg - average of all analyses results over bootstraps
                boot_std - standard deviation of all results over bootstraps
                boot_batch_std - for those analyses involving batching, retuns the standard
                                 deviation computed from uncertainy propagation of each of
                                 the batch variances from each bootstrap
        """
        #To bootstrap, just run _all_analyses_ multiple times and take statistics
        boots = []
        #Above records each bootstrapped estimate, below records variances in estimates for batched analyses
        boots_vars = []
        for i in range(n_boot):
            # this_inds = self.rng.choice(self.z_means.shape[0], size=self.z_means.shape[:1], replace=True)
            this_inds = self.rng.uniform((self.n_obs,), minval=0, maxval=self.n_obs, dtype=tf.int32)
            if i == 0:
                this_result = self._all_analyses_(indices=this_inds)
                boots = [[out] for out in this_result[0]]
                boots_vars = [[out] for out in this_result[1]]
            else:
                this_result = self._all_analyses_(indices=this_inds)
                for ind in range(len(boots)):
                    boots[ind].append(this_result[0][ind])
                for ind in range(len(boots_vars)):
                    boots_vars[ind].append(this_result[1][ind])
        boots = [np.array(out) for out in boots]
        boot_avg = [np.average(out, axis=0) for out in boots]
        boot_std = [np.std(out, axis=0, ddof=1) for out in boots]
        boot_batch_std = [np.sqrt(np.sum(out, axis=0))/n_boot for out in boots_vars]
        #Could also consider reporting standard error in bootstrapped mean (divide by sqrt(n_boot))
        #Or reporting percentiles (say 5 and 95) instead of std
        return boot_avg, boot_std, boot_batch_std

    def analyze(self, bootstrap=False, n_boot=100, n_per=1):
        """
        Performs analysis.

        Inputs:
                bootstrap - (False) bool if should perform bootstrapping or not
                n_boot - (100) number of bootstrap samples for bootstrapping
                n_per  - (1) when not bootstrapping, number of z samples per x to draw
        Outputs:
                Stores dictionary of results in self.results
        """
        #Perform analysis and store results in self.results dictionary
        if bootstrap:
            result = self._bootstrap_(n_boot=n_boot)
            self.results = {**self.results, **dict(zip(self.result_keys, result[0]))}
            self.results['uncertainty'] = {**self.results['uncertainty'], **dict(zip(self.result_keys, result[1]))}
            #Correct uncertainties for batched analyses to include batch variance
            #Do manually (don't see other clean way)
            self.results['uncertainty']['qzx_jd_per_dim'] = result[2][0]
            self.results['uncertainty']['qz_tot_corr_tcvae'] = result[2][1]
        else:
            result = self._all_analyses_(n_per=n_per)
            result_means = [out.numpy() for out in result[0]]
            result_vars = [out.numpy() for out in result[1]]
            self.results = {**self.results, **dict(zip(self.result_keys, result_means))}
            #Since have uncertainties just from batching, display those for batched analyses
            self.results['uncertainty']['qzx_jd_per_dim'] = np.sqrt(result_vars[0])
            self.results['uncertainty']['qz_tot_corr_tcvae'] = np.sqrt(result_vars[1])


