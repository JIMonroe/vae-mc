# Written by Jacob I. Monroe, NIST Employee

"""Defines methods to load data for training VAE models."""

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

from libVAE.coord_transforms import sincos


def make_tf_dataset(rawData, val_frac, batch_size, thermo_betas=None):
  """Generic function to turn loaded data into a tf Dataset object.
If thermo_betas are provided, will treat as reciprocal temperatures (1/kB*T)
and use as additional information for training the model, like "labels."
Otherwise, will just duplicate configs as the "targets."
  """
  #Save some fraction of data for validation
  valInd = int((1.0-val_frac)*rawData.shape[0])

  trainData = tf.data.Dataset.from_tensor_slices(rawData[:valInd])
  valData = tf.data.Dataset.from_tensor_slices(rawData[valInd:])

  if thermo_betas is not None:
    betaData = tf.data.Dataset.from_tensor_slices(thermo_betas[:valInd])
    valbetaData = tf.data.Dataset.from_tensor_slices(thermo_betas[valInd:])
    trainData = tf.data.Dataset.zip((trainData, betaData))
    valData = tf.data.Dataset.zip((valData, valbetaData))
  else:
    trainData = tf.data.Dataset.zip((trainData, trainData))
    valData = tf.data.Dataset.zip((valData, valData))

  trainData = trainData.shuffle(buffer_size=3*batch_size).batch(batch_size)
  valData = valData.batch(batch_size)

  return trainData, valData


def raw_image_data(datafile, file_betas=None, ref_beta=1.77):
  """Reads in data in netcdf format and retuns numpy array.
Really working with lattice gas snapshots, which we can consider images.
If given a list, will loop over.
And if passed file_betas, build list of reciprocal temperatures (1/kB*T) associated
with each configuration, assuming 1 temperature for each data file.
Reference beta is hard-coded, so watch out!
  """
  if isinstance(datafile, str):
    dat = Dataset(datafile, 'r')
    images = np.array(dat['config'][:,:,:], dtype='float32')
    dat.close()
    if file_betas is not None:
      betas = file_betas*np.ones(images.shape[0], dtype='float32')
  else:
    images = np.array([])
    if file_betas is not None:
      betas = np.array([])
    for i, f in enumerate(datafile):
      dat = Dataset(f, 'r')
      if i == 0:
        images = np.array(dat['config'][:,:,:], dtype='float32')
        if file_betas is not None:
          betas = file_betas[i]*np.ones(dat['config'].shape[0], dtype='float32')
      else:
        images = np.vstack((images, np.array(dat['config'][:,:,:], dtype='float32')))
        if file_betas is not None:
          betas = np.hstack((betas, file_betas[i]*np.ones(dat['config'].shape[0], dtype='float32')))
      dat.close()
  #Most images have 3 dimensions, so to fit with previous VAE code for images, add dimension
  images = np.reshape(images, images.shape+(1,))
  #And want to shuffle data randomly, but need to do data and betas in same way
  perm = np.random.permutation(images.shape[0])
  images = images[perm, ...]
  if file_betas is not None:
    #betas = ref_beta / betas[perm] #Divide ref_beta by betas so T/T0, so higher T has more std
    betas = np.exp(6.0*((betas[perm] / ref_beta) - 1.0)) #Need more drastic changes in std
    #BUT, for lattice gas, latent space has high temperatures in middle of distribution
    #So with broader P(z') for higher T, need transformation of z' ~ 1/z
    #Unfortunately, RQS is monotonically increasing transformation over transformed interval
    #So can never learn flow to map high T to higher std of P(z')
    #But can go the other way, though it's based on prior knowledge of learned encoding
    return images, betas
  else:
    return images


def image_data(datafile, batch_size, val_frac=0.1, file_betas=None):
  """Takes raw data as numpy array and converts to training and validation tensorflow datasets."""
  if file_betas is not None:
    rawData, thermoBeta = raw_image_data(datafile, file_betas=file_betas)
    return make_tf_dataset(rawData, val_frac, batch_size, thermo_betas=thermoBeta)
  else:
    rawData = raw_image_data(datafile)
    return make_tf_dataset(rawData, val_frac, batch_size)


def dsprites_data(batch_size, val_frac=0.01):
  """Loads in the dsprites dataset from tensorflow_datasets.
  """
  import tensorflow_datasets as tfds

  def imFromDict(adict):
    return tf.cast(adict['image'], 'float32')

  valPercent = int(val_frac*100)
  trainData = tfds.load("dsprites", split="train[:%i%%]"%(100-valPercent))
  trainData = trainData.map(imFromDict)
  trainData = trainData.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)
  trainData = tf.data.Dataset.zip((trainData, trainData))
  valData = tfds.load("dsprites", split="train[-%i%%:]"%(valPercent))
  valData = valData.map(imFromDict)
  valData = valData.batch(batch_size, drop_remainder=True)
  valData = tf.data.Dataset.zip((valData, valData))
  return trainData, valData


def dimer_2D_data(datafile, batch_size, val_frac=0.01,
                  dset='all', permute=True, center_and_whiten=False):
  """Loads data from a system of 2D LJ particles with one special dimer
generated by Frank Noe's deep_boltzmann package.
  """
  trajdict = np.load(datafile, allow_pickle=True)

  #Load open and closed states or just one of them
  if dset == 'all':
    trajStrs = ['traj_open', 'traj_closed']
  else:
    trajStrs = ['traj_%s'%dset]

  #Use permuted particle labelling if desired
  if permute:
    trajStrs = [s+'_hungarian' for s in trajStrs]

  rawData = np.vstack([trajdict[s] for s in trajStrs])
  #And want to shuffle data randomly (just a good idea)
  np.random.shuffle(rawData)

  #Center and whiten the data if set to
  #If train model with this, make sure to unwhiten before comparing
  if center_and_whiten:
    rawData -= np.mean(rawData, axis=0)
    rawData /= np.std(rawData, axis=0, ddof=1)

  #Re-order particles in a way that will work well with autoregression
  #Remember, data is x1, y1, x2, y2, etc.
  #Spiralling out from dimer
  #particle_order = [0, 1, #Dimer particles
  #             17, 16, 22, 23, #4 particles closest to dimer
  #             18, 11, 10, 15, 21, 28, 29, 24, #Next layer around dimer excluding corners
  #             12, 9, 27, 30, #Corners, which have less freedom
  #             31, 25, 19, 13, 6, 5, 4, 3, 8, 14, 20, 26, 33, 34, 35, 36, #Outer layer, no corners
  #             37, 7, 2, 32] #Outermost corners
  #new_order = np.zeros(rawData.shape[1], dtype=int)
  #new_order[::2] = np.array(particle_order)*2
  #new_order[1::2] = np.array(particle_order)*2 + 1
  #rawData = rawData[:, new_order]

  return make_tf_dataset(rawData, val_frac, batch_size)


def ala_dipeptide_data(datafile, batch_size, val_frac=0.01, rigid_bonds=False, sin_cos=False):
  """Loads data for alanine dipeptide. Can load either XYZ or BAT coordinates, but in
analysis, make sure to also use MDAnalysis.analysis.bat to switch between them if needed.
  """
  rawData = np.load(datafile).astype('float32')

  #If have rigid bonds (and have BAT coordinates so can constrain them in VAE model)
  #then want to identify them and mask them out during training
  #First 3 DOFs are root atom coordiantes, next three rotational coordinates
  #Next 2 are first 2 bonds, then have first angle
  totDOFs = rawData.shape[1]
  if 'BAT' in datafile and rigid_bonds:

    bond_inds = list(range(6)) #[6, 7] Masking rigid translation and rotation too
    #First two bonds don't involve hydrogens, so not constrained
    #Next have all bonds, with number being Natoms - 1
    #Even for cyclic compounds, this will be the case because we mean "independent DOF bonds"
    #But not all bonds are constrained... so pick these out manually
    #for b in range(9, 9 + totDOFs//3 - 3):
    #  bond_inds.append(b)
    bond_inds = bond_inds + [9, 11, 12, 13, 15, 17, 18, 19, 21, 24, 25, 26]
    bond_mask = np.ones(totDOFs, dtype=bool)
    bond_mask[bond_inds] = False
    rawData = rawData[:, bond_mask]

    if sin_cos:
      rawData = sincos(rawData, totDOFs)
      #Instead of dihedral angles, input sine-cosine pairs
      #Will have Natoms - 3 dihedral angles in BAT coordinates and will be at end
      #torsion_sin = np.sin(rawData[:, -(totDOFs//3 - 3):])
      #torsion_cos = np.cos(rawData[:, -(totDOFs//3 - 3):])
      #rawData = np.concatenate([rawData[:, :-(totDOFs//3 - 3)],
      #                         torsion_sin,
      #                         torsion_cos], axis=1)

  return make_tf_dataset(rawData, val_frac, batch_size)


def polymer_data(datafile, batch_size, val_frac=0.01, rigid_bonds=False, sin_cos=False):
  """Loads data for alanine dipeptide. Can load either XYZ or BAT coordinates, but in
analysis, make sure to also use MDAnalysis.analysis.bat to switch between them if needed.
If have rigid bonds, best to incorporate these constraints into model, which is naturally
accomplished with BAT coordinates and only passing DOFs that aren't bonds.
  """
  rawData = np.load(datafile).astype('float32')

  #If have rigid bonds (and have BAT coordinates so can constrain them for VAE model)
  #then want to identify them and mask them out in the training data
  #First 3 DOFs are root atom coordinates, next three are rotational coordinates
  #Next 2 are first two bonds, then we have the first angle
  totDOFs = rawData.shape[1]
  if 'BAT' in datafile and rigid_bonds:

    bond_inds = list(range(8)) #[6, 7] Masking rigid translation and rotation as well
    #Next have all bonds, with this number depending on number of atoms
    #For polymer, should just have one bead per monomer, so Nbonds = N-1
    for b in range(9, 9 + totDOFs//3 - 3):
      bond_inds.append(b)
    bond_mask = np.ones(totDOFs, dtype=bool)
    bond_mask[bond_inds] = False
    rawData = rawData[:, bond_mask]

    if sin_cos:
      rawData = sincos(rawData, totDOFs)
      #Input sine-cosine pairs instead of dihedral angles
      #With autoregressive it's easier to just work with those throughout
      #torsion_sin = np.sin(rawData[:, -(totDOFs//3 - 3):])
      #torsion_cos = np.cos(rawData[:, -(totDOFs//3 - 3):])
      #rawData = np.concatenate([rawData[:, :-(totDOFs//3 - 3)],
      #                         torsion_sin,
      #                         torsion_cos], axis=1)

    ##Also reorder so that autoregressive model predicts angle, dihedral, angle dihedral, etc.
    #dof_order = []
    #for a in range(totDOFs//3 - 2): #Number of angles
    #  dof_order.append(a)
    #  #Only add dihedral if have any left
    #  if a < totDOFs//3 - 3:
    #    dof_order.append(a + totDOFs//3 - 2)
    #rawData = rawData[:, dof_order]

  #If have no cyclic structures, above should work for any bond topography

  return make_tf_dataset(rawData, val_frac, batch_size)


