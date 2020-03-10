# Written in 2020 by Jacob I. Monroe, NIST Employee

"""Defines methods to load data for training VAE models."""

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

def raw_image_data(datafile):
  """Reads in data in netcdf format and retuns numpy array.
Really working with lattice gas snapshots, which we can consider images."""
  dat = Dataset(datafile)
  images = np.array(dat['config'][:,:,:], dtype='float32')
  dat.close()
  #Most images have 3 dimensions, so to fit with previous VAE code for images, add dimension
  images = np.reshape(images, images.shape+(1,))
  #And want to shuffle data randomly
  np.random.shuffle(images)
  return images


def image_data(datafile, batch_size, val_frac=0.1):
  """Takes raw data as numpy array and converts to training and validation tensorflow datasets."""
  rawData = raw_image_data(datafile)
  #Save some fraction of data for validation
  valInd = int((1.0-val_frac)*rawData.shape[0])
  trainData = tf.data.Dataset.from_tensor_slices(rawData[:valInd])
  trainData = trainData.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)
  trainData = tf.data.Dataset.zip((trainData, trainData))
  valData = tf.data.Dataset.from_tensor_slices(rawData[valInd:])
  valData = valData.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)
  valData = tf.data.Dataset.zip((valData, valData))
  return trainData, valData


def dsprites_data(batch_size, val_frac=0.1):
  """Loads in the dsprites dataset from tensorflow_datasets.
  """
  import tensorflow_datasets as tfds
  valPercent = int(val_frac*100)
  trainData = tfds.load("dsprites", split="train[:100000]")#split="train[:%%%i]"%(100-valPercent))
  trainData = tf.data.Dataset.from_tensor_slices([tf.cast(dat['image'], 'float32')
                                                 for dat in trainData])
  trainData = trainData.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)
  trainData = tf.data.Dataset.zip((trainData, trainData))
  valData = tfds.load("dsprites", split="train[-640:]")#split="train[-%%%i:]"%(valPercent))
  valData = tf.data.Dataset.from_tensor_slices([tf.cast(dat['image'], 'float32')
                                                for dat in valData])
  valData = valData.shuffle(buffer_size=batch_size).batch(batch_size, drop_remainder=True)
  valData = tf.data.Dataset.zip((valData, valData))
  return trainData, valData


