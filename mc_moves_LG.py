# Written by Jacob I. Monroe, NIST Employee

"""MC moves specific to the lattice gas and designed to efficiently run simulations
in parallel on GPUs with tensorflow
"""

import copy
import numpy as np
import tensorflow as tf

from libVAE import losses


def moveTranslate(currConfig, currU, B,
                  energy_func=losses.latticeGasHamiltonian,
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


def moveDeleteMulti(currConfig, currU, B,
                    energy_func=losses.latticeGasHamiltonian,
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


def moveInsertMulti(currConfig, currU, B,
                    energy_func=losses.latticeGasHamiltonian,
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


