
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from architectures import MaskedNet


def identity_transform(coords, reverse=False):
  """Identity transformation returning input coordinates.
  """
  return tf.identity(coords)


def spherical_transform(coords, dist_sq=None, reverse=False):
  """Transformation from Cartesian to spherical coordinates (forward) or vice versa (reverse).
  Inputs:
          coords - input coordinates (last dimension of 3); should be (x, y, z) or (r, azimuth, polar)
          dist_sq - (None) squared distances if already computed (for forward transformation)
          reverse - (False) whether to go from spherical to Cartesian
  Outputs:
          transformed coordinates - (r, azimuth, polar) for forward, (x, y, z) for reverse
  """
  if reverse:
    x = coords[..., 0]*tf.math.cos(coords[..., 1])*tf.math.sin(coords[..., 2])
    y = coords[..., 0]*tf.math.sin(coords[..., 1])*tf.math.sin(coords[..., 2])
    z = coords[..., 0]*tf.math.cos(coords[..., 2])
    return tf.stack([x, y, z], axis=-1)
  else:
    if dist_sq is not None:
      r = tf.sqrt(dist_sq)
    else:
      r = tf.math.reduce_euclidean_norm(coords, axis=-1)
    azimuth = tf.math.atan2(coords[..., 1], coords[..., 0])
    polar = tf.math.acos(coords[..., 2] / r)
    return tf.stack([r, azimuth, polar], axis=-1)


def distance_mask(refCoord, coords, k_neighbors=12):
  """Distance-based (Euclidean) mask to select k-nearest neighbors of a point.
  Inputs:
          refCoord - reference coordinate to define distances and neighbors
          coords - all other coordinates to consider
          k_neighbors - (12) number of nearest neighbors to select, masking other coords
  Outputs:
          k_coords - k nearest coordinates in terms of local coordinate from refCoord (coords - refCoord)
          k_inds - indices in the original coords of the k closest in k_coords
          k_dists - squared distances (Euclidean) from refCoord for k nearest neighbors
  """
  local_coords = coords - refCoord
  dist_sq = tf.reduce_sum(local_coords * local_coords, axis=-1)
  k_dists, k_inds = tf.math.top_k(-dist_sq, k=k_neighbors) #Takes k largest values, so negate
  k_coords = tf.gather(local_coords, k_inds, axis=-2, batch_dims=len(coords.shape)-2)
  return k_coords, k_inds, -k_dists


def create_dist(params, tfp_dist_list, param_transforms=None):
  """Create sampling distribution for autoregressive sampling.
  Inputs:
          params - array or tensor of dimensions (batch, particle, coordinates, params)
          tfp_dist_list - list of tfp distributions matching length of coordinates dimension
  Outputs:
          base_dist - joint tfp distribution over all batches, particles, and coordinates
  """
  #Set transform to default if None
  if param_transforms is None:
    param_transforms = [tf.identity]*params.shape[-1]
  base_dist_list = []
  #Loop over second to last dimension of params
  #(coordinates of each particle, last dimension is prob dist parameters for each coordinate)
  for i in range(params.shape[-2]):
    this_params = [param_transforms[k](params[..., i, k]) for k in range(params.shape[-1])]
    base_dist_list.append(tfp_dist_list[i](*this_params))
  #Put together into a joint distribution
  base_dist = tfp.distributions.JointDistributionSequential(base_dist_list)
  return base_dist


class SolvationNet(tf.keras.layers.Layer):
  """Network with autoregressive prediction of probability distribution parameters,
specifically for predicting new solvent coordinates given solute or solvent. If want to
include already placed solvent in prediction, set augment_input to True and provide the
keyword argument extra_coords when calling this class.
  """
  def __init__(self,
               out_shape,
               name='solv_net',
               out_event_dims=2, #Number parameters for probability distributions
               hidden_dim=50,
               n_hidden=3,
               augment_input=True,
               **kwargs):
    """
    Inputs:
            out_shape - shape of output (so for particles, [batch_dim, n_particles, n_dims]); note that actual output will have one added dimension of size out_event_dims
            name - ('solv_net', str) name of the layer
            out_event_dims - (2, int) number of probability parameters
            hidden_dim - (50, int) dimension of all hidden layers
            n_hidden - (3, int) number of hidden layers before predicting base parameters
            augment_input - (True, bool) whether or not to expect to seperate sets of input coordinates; if so, will add a second set of hidden layers to process extra input; useful for separating out and processing solute and solvent input coordinates differently
    Outputs:
             SolvationNet layer
    """
    super(SolvationNet, self).__init__(name=name, **kwargs)
    self.out_shape = out_shape
    self.out_event_dims = out_event_dims
    self.auto_groupings = np.hstack([[k]*out_shape[-1] for k in np.arange(1, out_shape[-2]+1)])
    self.hidden_dim = hidden_dim
    #Need function to flatten provided training data
    self.flat = tf.keras.layers.Flatten()
    #Generate hidden layers
    self.hidden_layers = []
    for i in range(n_hidden):
      self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_dim,
                                                      activation=tf.nn.relu))
    #And final layer for base parameters
    self.base_param = tf.keras.layers.Dense(self.out_event_dims*np.prod(self.out_shape),
                                            activation=None)
    #Also create hidden layers based on additional input coordinates (if desired)
    self.augment_input = augment_input
    if self.augment_input:
      self.augment_hidden = []
      for i in range(n_hidden):
        self.augment_hidden.append(tf.keras.layers.Dense(self.hidden_dim,
                                                         activation=tf.nn.relu))

  def build(self, input_shape):
    #Build autoregressive network only once input is defined
    #If want conditional input, need to know input dimension
    self.autonet = MaskedNet(self.out_event_dims,
                             event_shape=np.prod(self.out_shape),
                             conditional=True,
                             conditional_event_shape=input_shape[1:],
                             hidden_units=[self.hidden_dim,]*2,
                             dof_order=self.auto_groupings,
                             activation='tanh',
                             use_bias=True)

  def call(self, input_coords, extra_coords=None, sampled_input=None):
    """
    Inputs:
            input_coords - coordinates of main particles, such as solutes or nearby solvents, passed as input
            extra_coords - (None, array) coordinates of other class of inputs (like solvent if solutes are input_coords, or vice-versa)
            sampled_input - (None, array) coordinates of already sampled particle positions to influence autoregressive model; for instance, this might be solvent being predicted by the autoregressive portion of THIS network
    Outputs:
            params, shifts - (sampled_input=None) base parameters without autoregression, and shifts from autoregressive model
            shifts - (sampled_input not None) shifts from autoregression given sampled_input and conditional on original input_coords (and potentially also extra_coords)
    """
    #Assumes coordinates have already been selected and transformed appropriately
    #(i.e., we're taking inputs and producing outputs all in terms of a central particle)
    flattened = self.flat(input_coords)
    if self.augment_input and extra_coords is not None:
      #If including extra coords, like solvent, include in conditional input
      extra_flat = self.flat(extra_coords)
      cond_input = tf.concat([flattened, extra_flat], axis=-1)
    else:
      cond_input = flattened

    #In this case, we only want to apply autonet, nothing else
    #Note that this will only return shifts
    if sampled_input is not None:
      shifts = self.autonet(sampled_input, conditional_input=cond_input)
      shifts = tf.reshape(shifts, (-1,)+self.out_shape+(self.out_event_dims,))
      return shifts
    #Alternatively, need to generate base parameters before applying shifts
    else:
      hidden_out = tf.identity(flattened)
      for h_layer in self.hidden_layers:
        hidden_out = h_layer(hidden_out)
      #And if including extras, will apply those hidden layers and augment hidden_out
      if self.augment_input and extra_coords is not None:
        extra_hidden_out = tf.identity(extra_flat)
        for h_layer in self.augment_hidden:
          extra_hidden_out = h_layer(extra_hidden_out)
        hidden_out = tf.concat([hidden_out, extra_hidden_out], axis=-1)
      #Predict base parameters without autoregressive shifts
      params = self.base_param(hidden_out)
      #Really just applying conditional input here, nothing more (skip connections)
      #Should only be in this code block if have not sampled yet
      #(or have not yet started labeling input data points)
      shifts = self.autonet(tf.zeros((params.shape[0], np.prod(self.out_shape))),
                            conditional_input=cond_input)
      params = tf.reshape(params, (-1,)+self.out_shape+(self.out_event_dims,))
      shifts = tf.reshape(shifts, (-1,)+self.out_shape+(self.out_event_dims,))
      return params, shifts


class ParticleDecoder(tf.keras.layers.Layer):
  """Decodes from solute coordinates, reintroducing solvent.
  """
  def __init__(self,
               solute_shell_num,
               num_solvent_nets,
               coordinate_dimensionality=3,
               k_solute_neighbors=12,
               k_solvent_neighbors=12,
               name='decoder',
               tfp_dist_list=None,
               param_transforms=None,
               coord_transform=identity_transform,
               hidden_dim=50,
               **kwargs):
    super(ParticleDecoder, self).__init__(name=name, **kwargs)
    #Define output shape for solute networks - number shell particles by dimensionality
    self.solute_out_shape = (solute_shell_num, coordinate_dimensionality)
    #Number of neighbors to select in distance masks for solute and solvent
    self.k_solute_neighbors = k_solute_neighbors
    self.k_solvent_neighbors = k_solvent_neighbors
    #Define probability distributions for each coordinate dimension
    if tfp_dist_list is None:
      self.tfp_dist_list = [tfp.distributions.Normal]*coordinate_dimensionality
    #Define transformations on parameters (for instance, if predict log(var), take exp)
    if param_transforms is None:
      self.param_transforms = [tf.identity, lambda x: tf.math.exp(0.5*x)] #Defaults to mean and log(var)
    else:
      self.param_transforms = param_transforms
    #Define coordinate transformation
    self.coord_transform = coord_transform
    #And hidden dimension for all networks
    self.hidden_dim = hidden_dim
    #Create all the networks that will be needed
    self.base_solute_net = SolvationNet(self.solute_out_shape,
                                       out_event_dims=len(self.param_transforms),
                                       hidden_dim=self.hidden_dim,
                                       n_hidden=3,
                                       augment_input=False)
    self.solute_net = SolvationNet(self.solute_out_shape,
                                  out_event_dims=len(self.param_transforms),
                                  hidden_dim=self.hidden_dim,
                                  n_hidden=3,
                                  augment_input=True)
    #Solvent nets to be handled as list because number may vary
    #Each network will double the number of solvent particles present
    #(by adding 1 new particle for each current particle)
    self.solvent_nets = []
    for i in range(num_solvent_nets):
      self.solvent_nets.append(SolvationNet((1, coordinate_dimensionality),
                                           out_event_dims=len(self.param_transforms),
                                           hidden_dim=self.hidden_dim,
                                           n_hidden=3,
                                           augment_input=True))

  def get_log_probs(self, coords, params, ref_coords):
    #Transform to local, transformed coordinates
    local_coords = self.coord_transform(coords - ref_coords)
    dist = create_dist(params, self.tfp_dist_list, param_transforms=self.param_transforms)
    log_probs = dist.log_prob(tf.unstack(local_coords, axis=-1))
    return log_probs

  def call(self, solute_coords, train_data=None):
    #Keep track of output solvent coordinates
    #(must do throughout because autoregressive)
    solvent_out = None
    #Also output parameters
    params_out = None
    #But parameters in local, transformed coords, so also track reference coords for params
    ref_out = None
    #For each solute coordinate, loop over and add solvation shell
    for i in range(solute_coords.shape[1]):
      #Find nearest solute neighbors
      ref_coord = solute_coords[:, i:i+1, :] #[n_batch, n_particles, n_coordinate]
      other_coords = tf.gather(solute_coords,
                               indices=np.hstack([np.arange(i),
                                                  np.arange(i+1, solute_coords.shape[1])]),
                               axis=1)
      k_solute_coords, k_solute_inds, k_solute_dists = distance_mask(ref_coord,
                                                                     other_coords,
                                                                     k_neighbors=self.k_solute_neighbors)
      #Transform coordinates
      k_solute_coords = self.coord_transform(k_solute_coords)
      #Change behavior if first solute
      if i == 0:
        #Predict initial parameters and shifts
        params, shifts = self.base_solute_net(k_solute_coords)
      else:
        #Again initial params and shifts, but now need closest solvent as well
        k_solvent_coords, k_solvent_inds, k_solvent_dists = distance_mask(ref_coord,
                                                                          solvent_out,
                                                                          k_neighbors=self.k_solvent_neighbors)
        k_solvent_coords = self.coord_transform(k_solvent_coords)
        params, shifts = self.solute_net(k_solute_coords, extra_coords=k_solvent_coords)
      #Sample first particle in solvation shell
      dist = create_dist(params + shifts,
                         self.tfp_dist_list,
                         param_transforms=self.param_transforms)
      sample = tf.stack(dist.sample(), axis=-1)
      this_solvent_out = tf.identity(sample)
      #Loop over rest of solvation shell to make autoregressive
      for j in range(1, self.solute_net.out_shape[0]):
        if i == 0:
          shifts = self.base_solute_net(k_solute_coords, sampled_input=this_solvent_out)
        else:
          shifts = self.solute_net(k_solute_coords,
                                   extra_coords=k_solvent_coords,
                                   sampled_input=this_solvent_out)
        dist = create_dist(params + shifts,
                           self.tfp_dist_list,
                           param_transforms=self.param_transforms)
        sample = tf.stack(dist.sample(), axis=-1)
        #Leave items less than index j unchanged in this_solvent_out, update rest
        this_solvent_out = tf.concat([this_solvent_out[:, :j, :], sample[:, j:, :]], axis=1)
      #Before moving to next solute particle, update solvent coordinates
      #Make sure to transform out of local coordinate system
      if i == 0:
        solvent_out = ref_coord + self.coord_transform(this_solvent_out, reverse=True)
        params_out = params + shifts
        ref_out = tf.tile(ref_coord, (1, this_solvent_out.shape[1], 1))
      else:
        solvent_out = tf.concat([solvent_out,
                           ref_coord + self.coord_transform(this_solvent_out, reverse=True)],
                                 axis=1)
        params_out = tf.concat([params_out, params + shifts], axis=1)
        ref_out = tf.concat([ref_out,
                             tf.tile(ref_coord, (1, this_solvent_out.shape[1], 1))],
                            axis=1)

    #Now all solvation shells should be placed
    #Loop through solvent networks, applying to each solvent particle
    #Good news is that for each solvent, just predict 1 particle position
    #So no need for a second autoregressive loop!
    #Will be autoregressive by virture of fact that keep adding to solvent_out
    for solv_net in self.solvent_nets:
      curr_num_solv = solvent_out.shape[1]
      for i in range(curr_num_solv):
        #Get nearby solutes and solvents
        ref_coord = solvent_out[:, i:i+1, :]
        other_coords = tf.gather(solvent_out,
                                 indices=np.hstack([np.arange(i),
                                                    np.arange(i+1, solvent_out.shape[1])]),
                                 axis=1)
        k_solute_coords, k_solute_inds, k_solute_dists = distance_mask(ref_coord,
                                                                       solute_coords,
                                                                       k_neighbors=self.k_solute_neighbors)
        k_solvent_coords, k_solvent_inds, k_solvent_dists = distance_mask(ref_coord,
                                                                       other_coords,
                                                                       k_neighbors=self.k_solvent_neighbors)
        #Transform coordinates
        k_solute_coords = self.coord_transform(k_solute_coords)
        k_solvent_coords = self.coord_transform(k_solvent_coords)
        #Get parameters and shifts from network
        params, shifts = solv_net(k_solvent_coords, extra_coords=k_solute_coords)
        dist = create_dist(params + shifts,
                           self.tfp_dist_list,
                           param_transforms=self.param_transforms)
        sample = tf.stack(dist.sample(), axis=-1)
        #Add sample directly to solvent_out after transforming
        solvent_out = tf.concat([solvent_out,
                                 ref_coord + self.coord_transform(sample, reverse=True)],
                                 axis=1)
        params_out = tf.concat([params_out, params + shifts], axis=1)
        ref_out = tf.concat([ref_out, ref_coord], axis=1)

    return solvent_out, params_out, ref_out




