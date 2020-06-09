# Code for invertible mappings

import os
import time
import tensorflow as tf
import numpy as np

from libVAE import losses, dataloaders

from deep_boltzmann.models.particle_dimer import ParticleDimer


class TransformNet(tf.keras.layers.Layer):
  """'Scaling' or 'translation' neural net as part of invertible transformation.
  """

  def __init__(self, output_dim=None, net_dim=1200, name='transform_net',
               kernel_initializer='glorot_normal', activation='relu', **kwargs):
    super(TransformNet, self).__init__(name=name, **kwargs)
    self.output_dim = output_dim
    self.net_dim = net_dim
    self.kernel_initializer = kernel_initializer
    if activation == 'relu':
      self.activation = tf.nn.relu
    elif activation == 'tanh':
      self.activation = tf.nn.tanh
    else:
      print('Activation not recognized, setting to relu.')
      self.activation = tf.nn.relu

  def build(self, input_shape):
    if self.output_dim is None:
      #First dimension is number of samples and may be None
      self.output_dim = input_shape[1]
    self.e1 = tf.keras.layers.Dense(self.net_dim, activation=self.activation, name="e1",
                                    kernel_initializer=self.kernel_initializer)
    self.e2 = tf.keras.layers.Dense(self.net_dim, activation=self.activation, name="e2",
                                    kernel_initializer=self.kernel_initializer)
    self.e3 = tf.keras.layers.Dense(self.output_dim, activation=None,
                                    kernel_initializer=self.kernel_initializer)

  def call(self, input_tensor):
    e1_out = self.e1(input_tensor)
    e2_out = self.e2(e1_out)
    out = self.e3(e2_out)
    return out


class SplitLayer(tf.keras.layers.Layer):
  """Layer to split (in half) and flatten raw input.
     By default the split_mask is set as alternating 1's and 0's after flattening.
     To change this behavior, provide any numpy array of 1's and 0's instead. A single
     tensor with its first dimension being 2 will be returned. If the merge() method
     is called instead of calling this class directly with the call() method, a tensor
     with its first dimension of 2 will be put back together.
  """

  def __init__(self, split_mask=None,
               name='split_layer', **kwargs):
    super(SplitLayer, self).__init__(name=name, **kwargs)
    self.split_mask = split_mask
    #Create flattening layer
    self.flat = tf.keras.layers.Flatten()

  def build(self, input_shape):
    #Store the input_shape to reshape output back to during merge
    #First dimension is the number of samples and can be None, so ignore
    self.out_shape = input_shape[1:]
    if self.split_mask is None:
      #Splitting flattened tensor, so take product of input_shape
      self.split_mask = np.zeros(np.prod(input_shape[1:]), dtype=int)
      self.split_mask[::2] = 1 #Alternating 1's and 0's by default
    #Will also define indices for merging back together
    #Note that dynamic_partition treats 0 as true, 1 as false, so invert mask here
    merge_list = tf.dynamic_partition(tf.range(np.prod(input_shape[1:])), 1-self.split_mask, 2)
    self.merge_inds = tf.argsort(tf.concat(merge_list, axis=0))

  def call(self, input_tensor, merge=False):
    """Performs split of an input tensor, flattening then splitting in half.
       If "merge" is set to True...
       merges two tensors back together given a tensor of dimension 2 (second dimension as
       first is always batch size).
       Also reshapes back to the original dimension.
    """
    if not merge:
      #First flatten the input, then split apart - tf.keras.layers.Flatten flattens all but dim 0
      flat_out = self.flat(input_tensor)
      #Apply mask as boolean mask to the second dimension (flattened)
      t1 = tf.boolean_mask(flat_out, self.split_mask, axis=1)
      t2 = tf.boolean_mask(flat_out, 1-self.split_mask, axis=1)
      out = tf.stack([t1, t2], axis=1)
      return out
    else:
      concat_out = tf.concat(tf.unstack(input_tensor, axis=1), axis=1)
      sorted_out = tf.gather(concat_out, self.merge_inds, axis=1)
      out = tf.reshape(sorted_out, (input_tensor.shape[0],)+self.out_shape)
      return out


class TransformBlock(tf.keras.layers.Layer):
  """Single transformation block of an invertible transformation.
     The call method proceeds in the forward direction, while using the reverse method
     will perform the transform in reverse. For this to work, the second input dimension
     must be 2 - in other words, the data should already be split. The first input
     dimension will of course be the number of samples. And the last dimension will
     be the dimension of the flattened data, cut in half (should flatten and split
     with the SplitLayer first).
  """

  def __init__(self, output_dim=None, net_dim=1200,
               name='transform_block', **kwargs):
    super(TransformBlock, self).__init__(name=name, **kwargs)
    self.output_dim = None
    self.net_dim = net_dim
    #Does the following work? - Works great! Note, though, that stores only most recent.
    self.log_det_for_val = None
    self.log_det_rev_val = None

  def build(self, input_shape):
    if self.output_dim is None:
      self.output_dim = input_shape[2]
    self.snet = TransformNet(output_dim=self.output_dim, net_dim=self.net_dim, activation='tanh')
    self.tnet = TransformNet(output_dim=self.output_dim, net_dim=self.net_dim, activation='relu')

  def call(self, input_tensor, reverse=False):
    """Forward transformation block if "reverse" is set to False (default).
       "reverse" enables calling this block in reverse using same neural nets.
    """
    if not reverse:
      s_out = self.snet(input_tensor[:, 0, :])
      self.log_det_for_val = tf.reduce_sum(s_out, axis=1)
      t_out = self.tnet(input_tensor[:, 0, :])
      trans_out = input_tensor[:, 1, :]*tf.math.exp(s_out) + t_out
      out = tf.stack([input_tensor[:, 0, :], trans_out], axis=1)
      return out
    else:
      s_out = self.snet(input_tensor[:, 0, :])
      self.log_det_rev_val = -tf.reduce_sum(s_out, axis=1)
      t_out = self.tnet(input_tensor[:, 0, :])
      trans_out = (input_tensor[:, 1, :] - t_out)*tf.math.exp(-s_out)
      out = tf.stack([input_tensor[:, 0, :], trans_out], axis=1)
      return out

  def log_det_for(self, input_tensor):
    """Computes the log of the Jacobian determinant in the forward direction.
    """
    return tf.reduce_sum(self.snet(input_tensor[:, 0, :]), axis=1)

  def log_det_rev(self, input_tensor):
    """Computes the log of the Jacobian determinant in the reverse direction.
    """
    return -tf.reduce_sum(self.snet(input_tensor[:, 0, :]), axis=1)


#Not currently hierarchical - will work on that later
class InvNet(tf.keras.Model):
  """Invertible transformation model.
  """

  def __init__(self, data_shape,
               net_dim=1200,
               n_blocks=4,
               split_mask=None,
               activation=None,
               name='invnet', **kwargs):
    super(InvNet, self).__init__(name=name, **kwargs)
    #Require data_shape so can know dimension of latent space without data
    self.data_shape = data_shape
    self.net_dim = net_dim
    self.split_mask = split_mask
    if activation not in [None, 'logits']:
      print('Activation not recognized, setting to None.')
      self.activation = None
    else:
      self.activation = activation
    #Define all of the layers we will need - remember, they all work in reverse, too
    self.splitter = SplitLayer(split_mask=self.split_mask)
    self.n_blocks = n_blocks #Should have at least 2 so transform everything
    self.block_list = []
    for i in range(self.n_blocks):
      self.block_list.append(TransformBlock(output_dim=np.prod(self.data_shape),
                                            net_dim=self.net_dim))
    #Again, will this work? Seems to - can also implement as a loss if we wanted
    self.log_det_for_sum = None
    self.log_det_rev_sum = None

  def call(self, inputs, reverse=False):
    if not reverse:
      #Before doing anything else, apply inverse activation
      if self.activation == 'logits':
        #Implement inverse of sigmoid, or logistic function
        #But do carefully to avoid inf or -inf
        #clip_out = tf.where((inputs==1.0), 1.0-1e-07, 1e-07)
        clip_out = tf.clip_by_value(inputs, 1e-07, 1.0-1e-07)
        act_out = tf.math.log(clip_out / (1.0 - clip_out))
      elif self.activation is None:
        act_out = inputs
      #First we flatten and split (after activation)
      split_out = self.splitter(act_out)
      #Next pass through each transformation block
      b_out = self.block_list[0](split_out)
      for block in self.block_list[1:]:
        b_out = block(b_out[:, ::-1, :]) #Flip what gets transformed for each block
      #Finally merge back together into original shape
      z_out = self.splitter(b_out, merge=True)
      #Also track log determinant
      #Can alternatively use tf.GradientTape() to watch the input and determine
      #the Jacobian through the batch_jacobian(outputs, inputs) function
      #Probably only easier once the Jacobian is really complicated for the activation
      self.log_det_for_sum = tf.reduce_sum([b.log_det_for_val for b in self.block_list], axis=0)
      if self.activation == 'logits':
        #Add sum of log of derivative of inverse sigmoid to the Jacobian determinant
        self.log_det_for_sum -= tf.reduce_sum(tf.math.log(inputs*(1.0 - inputs)),
                                              axis=tf.range(1, len(z_out.shape)))
      return z_out
    else:
      #Reverse previous procedure
      split_out = self.splitter(inputs)
      b_out = self.block_list[-1](split_out, reverse=True)
      for block in self.block_list[-2::-1]:
        b_out = block(b_out[:, ::-1, :], reverse=True)
      x_out = self.splitter(b_out, merge=True)
      #Also track log determinant
      self.log_det_rev_sum = tf.reduce_sum([b.log_det_rev_val for b in self.block_list], axis=0)
      if self.activation == 'logits':
        #Add sum of log of derivative of sigmoid to the Jacobian determinant
        self.log_det_rev_sum += tf.reduce_sum(tf.math.log_sigmoid(x_out)
                                              + tf.math.log_sigmoid(-x_out),
                                              axis=tf.range(1, len(x_out.shape)))
        x_out = tf.math.sigmoid(x_out)
      return x_out


def linlogcut_tf(x, high_E=100, max_E=1e10):
    """Function to clip large energies - taken from Frank Noe's deep_boltzmann package.
    """
    # cutoff x after max_E - this should also cutoff infinities
    x = tf.where(x < max_E, x, max_E * tf.ones(tf.shape(x)))
    # log after high_E
    y = high_E + tf.where(x < high_E, x - high_E, tf.math.log(x - high_E + 1))
    # make sure everything is finite
    y = tf.where(tf.math.is_finite(y), y, max_E * tf.ones(tf.shape(y)))
    return y


def trainFromLatent(model,
                    num_steps=10000,
                    batch_size=100,
                    save_dir='inv_info',
                    overwrite=False,
                    beta=2.0,
                    energy_params={'mu':-2.0, 'eps':-1.0}):
  """Trains an invertible model by sampling from the latent distribution and computing
     the loss function derived by Noe, et al. (2019). The latent distribution is a standard
     normal distribution (zero mean and unit standard deviation).
  """

  #Set up checkpointing and saving - load previous model parameters if we can
  if os.path.isdir(save_dir):
    #If overwrite is True, don't need to do anything
    #If it's False, create a new directory to save to
    if not overwrite:
      print("Found saved model at %s and overwrite is False."%save_dir)
      print("Will attempt to load and continue training.")
      model.load_weights(os.path.join(save_dir, 'training.ckpt'))
      try:
        dir_split = save_dir.split('_')
        train_num = int(dir_split[-1])
        save_dir = '%s_%i'%("_".join(dir_split[:-1]), train_num+1)
      except ValueError:
        save_dir = '%s_1'%(save_dir)
      os.mkdir(save_dir)
  #If not, create that directory to save to later
  else:
    os.mkdir(save_dir)

  print("Model set up and ready to train.")

  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Set up optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  print("Beginning training at: %s"%time.ctime())

  pot_energy = losses.latticeGasHamiltonian
  #params = ParticleDimer.params_default.copy()
  #params['dimer_slope'] = 2.0
  #dimer_model = ParticleDimer(params=params)
  #pot_energy = dimer_model.energy_tf

  #Will loop over num_steps, creating sample of size batch_size each time for training
  #Loss will be part of training loop
  for step in range(num_steps):

    #Draw from standard normal
    z_sample = tf.random.normal((batch_size,)+model.data_shape)

    #Need to do all of the next bit within gradient_tape
    with tf.GradientTape() as tape:
      #Convert latent space representation into real-space
      #If working with lattice gas model, make sure to pass through sigmoid function
      #(that way bounded from 0 to 1 and interpret as probability)
      #Do this by setting activation='logits' in model because have to handle Jacobian there
      x_configs = model(z_sample, reverse=True)
      #Calculate the potential energy of the configurations
      u_vals = beta*pot_energy(x_configs, **energy_params)
      u_vals_clipped = linlogcut_tf(u_vals, high_E=10000, max_E=1e10)
      #And calculate total loss
      loss_energy = tf.reduce_mean(u_vals_clipped)
      loss_jacobian = tf.reduce_mean(model.log_det_rev_sum)
      loss = loss_energy - loss_jacobian

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if step%1000 == 0:
      print('\tStep %i: loss=%f, log_probs=%f, logRzx=%f'%(step, loss,
                                                           loss_energy, loss_jacobian))
      #Save checkpoint
      print('\tSaving checkpoint.')
      model.save_weights(checkpoint_path.format(epoch=step))

  print("Training completed at: %s"%time.ctime())
  model.save_weights(checkpoint_path.format(epoch=num_steps-1))
  print(model.summary())


def trainFromExample(model,
                     data_file,
                     num_epochs=2,
                     batch_size=64,
                     save_dir='inv_info',
                     overwrite=False):
  """Trains an invertible model by examples provided in data_file.
  """

  #Set up checkpointing and saving - load previous model parameters if we can
  if os.path.isdir(save_dir):
    #If overwrite is True, don't need to do anything
    #If it's False, create a new directory to save to
    if not overwrite:
      print("Found saved model at %s and overwrite is False."%save_dir)
      print("Will attempt to load and continue training.")
      model.load_weights(os.path.join(save__dir, 'training.ckpt'))
      try:
        dir_split = save_dir.split('_')
        train_num = int(dir_split[-1])
        save_dir = '%s_%i'%("_".join(dir_split[:-1]), train_num+1)
      except ValueError:
        save_dir = '%s_1'%(save_dir)
      os.mkdir(save_dir)
  #If not, create that directory to save to later
  else:
    os.mkdir(save_dir)

  print("Model set up and ready to train.")

  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Load in data
  trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.05)
  #trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
  #                                               dset='all', permute=True)

  #Set up optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  print("Beginning training at: %s"%time.ctime())

  #Will loop over epochs 
  for epoch in range(num_epochs):
    print('\nOn epoch %d:'%epoch)

    #Iterate over batches in the dataset
    for step, x_batch_train in enumerate(trainData):
      #Need to do all of the next bit within gradient_tape
      with tf.GradientTape() as tape:
        #Convert real space to latent space representation
        #Sticking to standard normal distributions in latent space
        z_configs = model(x_batch_train[0])
        #Calculate the log probabilities in latent space given standard normal distribution
        log_probs = tf.reduce_sum(tf.square(z_configs), axis=tf.range(1, len(z_configs.shape)))
        #And calculate total loss
        loss_probs = tf.reduce_mean(log_probs)
        loss_jacobian = tf.reduce_mean(model.log_det_for_sum)
        loss = loss_probs - loss_jacobian

      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step%100 == 0:
        print('\tStep %i: loss=%f, log_probs=%f, logRxz=%f'%(step, loss,
                                                             loss_probs, loss_jacobian))

    #Save checkpoint
    print('\tEpoch finished, saving checkpoint.')
    model.save_weights(checkpoint_path.format(epoch=epoch))

    #Check against validation data
    val_loss = tf.constant(0.0)
    batchCount = 0.0
    for x_batch_val in valData:
      z_configs = model(x_batch_val[0])
      log_probs = tf.reduce_sum(tf.square(z_configs), axis=tf.range(1, len(z_configs.shape)))
      val_loss += tf.reduce_mean(log_probs - model.log_det_for_sum)
      batchCount += 1.0
    val_loss /= batchCount
    print('\tValidation loss=%f'%(val_loss))

  print("Training completed at: %s"%time.ctime())
  print(model.summary())


def trainWeighted(model,
                  data_file,
                  num_epochs=2,
                  batch_size=64,
                  save_dir='inv_info',
                  overwrite=False,
                  beta=2.0,
                  energy_params={'mu':-2.0, 'eps':-1.0},
                  loss_weights=[1.0, 1.0]):
  """Trains an invertible model by examples provided in data_file.
  """

  #Set up checkpointing and saving - load previous model parameters if we can
  if os.path.isdir(save_dir):
    #If overwrite is True, don't need to do anything
    #If it's False, create a new directory to save to
    if not overwrite:
      print("Found saved model at %s and overwrite is False."%save_dir)
      print("Will attempt to load and continue training.")
      model.load_weights(os.path.join(save__dir, 'training.ckpt'))
      try:
        dir_split = save_dir.split('_')
        train_num = int(dir_split[-1])
        save_dir = '%s_%i'%("_".join(dir_split[:-1]), train_num+1)
      except ValueError:
        save_dir = '%s_1'%(save_dir)
      os.mkdir(save_dir)
  #If not, create that directory to save to later
  else:
    os.mkdir(save_dir)

  print("Model set up and ready to train.")

  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Load in data
  trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.05)
  #trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
  #                                               dset='all', permute=True)

  #Set up optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  print("Beginning training at: %s"%time.ctime())

  pot_energy = losses.latticeGasHamiltonian
  #params = ParticleDimer.params_default.copy()
  #params['dimer_slope'] = 2.0
  #dimer_model = ParticleDimer(params=params)
  #pot_energy = dimer_model.energy_tf

  #Will loop over epochs 
  for epoch in range(num_epochs):
    print('\nOn epoch %d:'%epoch)

    #Iterate over batches in the dataset
    for step, x_batch_train in enumerate(trainData):

      #For each x batch, also generate random z batch
      #Draw from standard normal
      z_sample = tf.random.normal((batch_size,)+model.data_shape)

      #Need to do all of the next bit within gradient_tape
      with tf.GradientTape() as tape:
        #First calculating loss through training by example
        #Convert real space to latent space representation
        #Sticking to standard normal distributions in latent space
        z_configs = model(x_batch_train[0])
        #Calculate the log probabilities in latent space given standard normal distribution
        log_probs = tf.reduce_sum(tf.square(z_configs), axis=tf.range(1, len(z_configs.shape)))
        #And calculate total loss
        loss_probs = tf.reduce_mean(log_probs)
        loss_jacobian_ex = tf.reduce_mean(model.log_det_for_sum)
        loss_ex = loss_probs - loss_jacobian_ex

        #Next get loss from training by sampling the latent space
        #Convert latent space representation into real-space
        #If working with lattice gas model, make sure to pass through sigmoid function
        #(that way bounded from 0 to 1 and interpret as probability)
        #Do this by setting activation='logits' in model because have to handle Jacobian there
        x_configs = model(z_sample, reverse=True)
        #Calculate the potential energy of the configurations
        u_vals = beta*pot_energy(x_configs, **energy_params)
        u_vals_clipped = linlogcut_tf(u_vals, high_E=10000, max_E=1e10)
        #And calculate total loss
        loss_energy = tf.reduce_mean(u_vals_clipped)
        loss_jacobian_la = tf.reduce_mean(model.log_det_rev_sum)
        loss_la = loss_energy - loss_jacobian_la

        #And total loss from both training by example and latent sampling
        loss = loss_weights[0]*loss_ex + loss_weights[1]*loss_la

      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step%100 == 0:
        print('\tStep %i: loss=%f, loss_ex=%f, log_probs=%f, logRxz=%f,' \
              ' loss_la=%f, log_energy=%f, logRzx=%f'%(step, loss,
                                                       loss_ex, loss_probs, loss_jacobian_ex,
                                                       loss_la, loss_energy, loss_jacobian_la))

    #Save checkpoint
    print('\tEpoch finished, saving checkpoint.')
    model.save_weights(checkpoint_path.format(epoch=epoch))

    #Check against validation data without sampling the latent space
    val_loss = tf.constant(0.0)
    batchCount = 0.0
    for x_batch_val in valData:
      z_configs = model(x_batch_val[0])
      log_probs = tf.reduce_sum(tf.square(z_configs), axis=tf.range(1, len(z_configs.shape)))
      val_loss += tf.reduce_mean(log_probs - model.log_det_for_sum)
      batchCount += 1.0
    val_loss /= batchCount
    print('\tValidation loss (example only) = %f'%(val_loss))

  print("Training completed at: %s"%time.ctime())
  print(model.summary())


