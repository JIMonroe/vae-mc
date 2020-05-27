# Code for invertible mappings

import os
import time
import tensorflow as tf
import numpy as np

from libVAE import losses


class TransformNet(tf.keras.layers.Layer):
  """'Scaling' or 'translation' neural net as part of invertible transformation.
  """

  def __init__(self, output_dim=None, net_dim=1200, name='transform_net',
               kernel_initializer='glorot_uniform', **kwargs):
    super(TransformNet, self).__init__(name=name, **kwargs)
    self.output_dim = output_dim
    self.net_dim = net_dim
    self.kernel_initializer = kernel_initializer

  def build(self, input_shape):
    if self.output_dim is None:
      #First dimension is number of samples and may be None
      self.output_dim = input_shape[1]
    self.e1 = tf.keras.layers.Dense(self.net_dim, activation=tf.nn.relu, name="e1",
                                    kernel_initializer=self.kernel_initializer)
    self.e2 = tf.keras.layers.Dense(self.net_dim, activation=tf.nn.relu, name="e2",
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

  def call(self, input_tensor):
    #First flatten the input, then split apart - tf.keras.layers.Flatten flattens all but dim 0
    flat_out = self.flat(input_tensor)
    #Apply mask as boolean mask to the second dimension (flattened)
    t1 = tf.boolean_mask(flat_out, self.split_mask, axis=1)
    t2 = tf.boolean_mask(flat_out, 1-self.split_mask, axis=1)
    out = tf.stack([t1, t2], axis=1)
    return out

  def merge(self, input_tensor):
    """Merges two tensors back together given a tensor of dimension 2 (second dimension as
       first is always batch size).
       Also reshapes back to the original dimension.
    """
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
    self.snet = TransformNet(output_dim=self.output_dim, net_dim=self.net_dim)
    self.tnet = TransformNet(output_dim=self.output_dim, net_dim=self.net_dim)

  def call(self, input_tensor):
    s_out = self.snet(input_tensor[:, 0, :])
    self.log_det_for_val = tf.reduce_sum(s_out, axis=1)
    t_out = self.tnet(input_tensor[:, 0, :])
    trans_out = input_tensor[:, 1, :]*tf.math.exp(s_out) + t_out
    out = tf.stack([input_tensor[:, 0, :], trans_out], axis=1)
    return out

  def reverse(self, input_tensor):
    """Want to be able to call this block in reverse using same neural nets
    """
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

  def __init__(self, data_shape, net_dim=1200, n_blocks=4, split_mask=None,
               name='invnet', **kwargs):
    super(InvNet, self).__init__(name=name, **kwargs)
    #Require data_shape so can know dimension of latent space without data
    self.data_shape = data_shape
    self.net_dim = net_dim
    self.split_mask = split_mask
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

  def call(self, inputs):
    #First we flatten and split
    split_out = self.splitter(inputs)
    #Next pass through each transformation block
    b_out = self.block_list[0](split_out)
    for block in self.block_list[1:]:
      b_out = block(b_out[:, ::-1, :]) #Flip what gets transformed for each block
    #Finally merge back together into original shape
    z_out = self.splitter.merge(b_out)
    #Also track log determinant
    self.log_det_for_sum = tf.reduce_sum([b.log_det_for_val for b in self.block_list], axis=0)
    return z_out

  def reverse(self, inputs):
    #Reverse previous procedure
    split_out = self.splitter(inputs)
    b_out = self.block_list[-1].reverse(split_out)
    for block in self.block_list[-2::-1]:
      b_out = block.reverse(b_out[:, ::-1, :])
    x_out = self.splitter.merge(b_out)
    #Also track log determinant
    self.log_det_rev_sum = tf.reduce_sum([b.log_det_rev_val for b in self.block_list], axis=0)
    return x_out


def trainFromLatent(model,
                    num_steps=10000,
                    batch_size=64,
                    save_dir='inv_info',
                    overwrite=False):
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
      print("Will attempt to load and ocntinue training.")
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

  #Set up optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  print("Beginning training at: %s"%time.ctime())

  #Will loop over num_steps, creating sample of size batch_size each time for training
  #Loss will be part of training loop
  for step in range(num_steps):

    #Draw from standard normal
    z_sample = tf.random.normal((batch_size, model.data_shape))

    #Need to do all of the next bit within gradient_tape
    with tf.GradientTape() as tape:
      #Convert latent space representation into real-space
      x_configs = model.reverse(z_sample)
      #Calculate the potential energy of the configurations
      u_vals = losses.LatticeGasHamiltonian(x_configs)
      #And calculate total loss
      loss = tf.reduce_mean(u_vals - model.log_det_rev_sum)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if step%1000 == 0:
      print('\nStep %i: loss=%f'%(step, loss))
      #Save checkpoint
      print('\tSaving checkpoint.')
      model.save_weights(checkpoint_path.format(epoch=step))

  print("Training completed at: %s"%time.ctime())
  model.save_weights(checkpoint_path.format(epoch=num_steps-1))
  print(model.summary())


