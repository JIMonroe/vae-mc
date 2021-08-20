
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from architectures import MaskedNet, NormFlowRealNVP, NormFlowRQSplineRealNVP


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
    polar = tf.math.acos(tf.math.divide_no_nan(coords[..., 2], r))
    return tf.stack([r, azimuth, polar], axis=-1)


class DistanceMask(object):
  """Class to define a callable operation that masks data by distance to a reference
coordinate. Useful as class rather than function because if have periodic simulation box,
need knowledge of the simulation box to correctly compute distances. This information
is stored as part of the class instance.
  """

  def __init__(self, box_lengths=None):
    """Creates instance of DistanceMask class
    Inputs:
            box_lengths - (None) simulation box edge lengths; if provided, applies periodic wrapping
    Outputs:
            DistanceMask object
    """
    #Need ability to apply periodic boundary conditions to compute distances
    self.boxL = box_lengths
    if self.boxL is None:
      self.periodic = False
    else:
      self.periodic = True

  def __call__(self, refCoord, coords, k_neighbors=12, ref_included=False):
    """Distance-based (Euclidean) mask to select k-nearest neighbors of a point.
    Inputs:
            refCoord - reference coordinate to define distances and neighbors
            coords - all other coordinates to consider
            k_neighbors - (12) number of nearest neighbors to select, masking other coords
            ref_included - (False) whether or not reference included in coords (will discard if True)
    Outputs:
            k_coords - k nearest coordinates in terms of local coordinate from refCoord (coords - refCoord)
            k_inds - indices in the original coords of the k closest in k_coords
            k_dists - squared distances (Euclidean) from refCoord for k nearest neighbors
    """
    if ref_included:
      k_neighbors = k_neighbors + 1
    local_coords = coords - refCoord
    if self.periodic:
      local_coords = local_coords - self.boxL*tf.round(local_coords / self.boxL)
    dist_sq = tf.reduce_sum(local_coords * local_coords, axis=-1)
    k_dists, k_inds = tf.math.top_k(-dist_sq, k=k_neighbors) #Takes k largest values, so negate
    if ref_included:
      k_dists = k_dists[:, 1:]
      k_inds = k_inds[:, 1:]
    k_coords = tf.gather(local_coords, k_inds, axis=-2, batch_dims=len(coords.shape)-2)
    return k_coords, k_inds, -k_dists


def create_dist(params, tfp_dist_list, param_transforms):
  """Create sampling distribution for autoregressive sampling.
  Inputs:
          params - array or tensor of dimensions (batch, particle, coordinates, params)
          tfp_dist_list - list of tfp distributions matching length of coordinates dimension
          param_transforms - list of transformations to apply to parameters before using to define a distribution; should be of same length as coordinates dimension and take the 
  Outputs:
          base_dist - joint tfp distribution over all batches, particles, and coordinates
  """
  base_dist_list = []
  #Loop over second to last dimension of params
  #(coordinates of each particle, last dimension is prob dist parameters for each coordinate)
  for i in range(params.shape[-2]):
    this_params = tf.unstack(params[..., i, :], axis=-1)
    this_params = param_transforms[i](*this_params)
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
            augment_input - (True, bool) whether or not to expect two seperate sets of input coordinates; if so, will add a second set of hidden layers to process extra input; useful for separating out and processing solute and solvent input coordinates differently
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
    batch_dim = tf.shape(input_coords)[0]
    #Assumes coordinates have already been selected and transformed appropriately
    #(i.e., we're taking inputs and producing outputs all in terms of a central particle)
    flattened = self.flat(input_coords)
    if self.augment_input and (extra_coords is not None):
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
      if self.augment_input and (extra_coords is not None):
        extra_hidden_out = tf.identity(extra_flat)
        for h_layer in self.augment_hidden:
          extra_hidden_out = h_layer(extra_hidden_out)
        hidden_out = tf.concat([hidden_out, extra_hidden_out], axis=-1)
      #Predict base parameters without autoregressive shifts
      params = self.base_param(hidden_out)
      #Really just applying conditional input here, nothing more (skip connections)
      #Should only be in this code block if have not sampled yet
      #(or have not yet started labeling input data points)
      shifts = self.autonet(tf.zeros((batch_dim, tf.reduce_prod(self.out_shape))),
                            conditional_input=cond_input)
      params = tf.reshape(params, (-1,)+self.out_shape+(self.out_event_dims,))
      shifts = tf.reshape(shifts, (-1,)+self.out_shape+(self.out_event_dims,))
      return params, shifts


class ParticleDecoder(tf.keras.layers.Layer):
  """Base class for particle decoders defining basic init and utility methods.
  """
  def __init__(self,
               coordinate_dimensionality=3,
               box_lengths=None,
               k_solute_neighbors=12,
               k_solvent_neighbors=12,
               tfp_dist_list=None,
               num_params=2,
               param_transforms=None,
               mean_transforms=None,
               coord_transform=identity_transform,
               hidden_dim=50,
               name='decoder',
               **kwargs):
    """Creates a base particle decoder class not intended to be used for anything, just helpful in defining solute and solvent decoding classes through inheritance. So you will need to implement a 'call' function if you inherit this class.
    Inputs:
            coordinate_dimensionality - (3) dimensionality of coordinates of particles
            box_lengths - (None) vector of length coordinate_dimensionality specifying box edge lengths if have periodic box and want to calculate distances taking that into consideration
            k_solute_neighbors - (12) number of nearest solute neighbors to consider
            k_solvent_neighbors - (12) number nearest solvent neighbors to consider when providing inputs to neural networks; effectively sets the n-body level at which autoregression is applied
            tfp_dist_list - (None) list of length coordinate_dimensionality that specifies tfp distributions that will have parameters provided to them by neural nets and drawn from for sampling; if None, default is to use normal distributions for all dimensions
            num_params - (2) number of parameters for tfp distributions
            param_transforms - (None) list of length coordinate_dimensionality specifying the transformations of outputs of the neural nets to pass as parameters to the tfp distributions; if None, default is to specify for normal distribution, so `lambda x, y: [x, tf.math.exp(0.5*y)]*coordinate_dimensionality` which assumes networks output mean and log(var)
            mean_transforms - (None) list of length coordinate_dimensionality to describe how the actual distribution mean is related to the neural network outputs; by default this uses tf.identity, but for something like spherical coordinates, the radial dimension requires postive means, so may need to have networks produce the log(mean) and apply an exponential
            coord_transform - (identity_transform) coordinate transformation in local frame around reference particle; by default, no transformation, but if provide function for this, should take cartesian coordinates as input and also have 'reverse' keyword argument to perform reverse transformation
            hidden_dim - (50) the number of units in hidden layers; probably want to be as large or larger than the number of output coordinates for any given network
    Outputs:
            ParticleDecoder object
    """
    super(ParticleDecoder, self).__init__(name=name, **kwargs)
    #Set dimensionality, which should match input!
    self.coord_dim = coordinate_dimensionality
    #Define distance mask with option to work directly with periodic simulation box
    self.distance_mask = DistanceMask(box_lengths=box_lengths)
    #Number of neighbors to select in distance masks for solute and solvent
    self.k_solute_neighbors = k_solute_neighbors
    self.k_solvent_neighbors = k_solvent_neighbors
    #Define probability distributions for each coordinate dimension
    if tfp_dist_list is None:
      self.tfp_dist_list = [tfp.distributions.Normal]*coordinate_dimensionality
    else:
      self.tfp_dist_list = tfp_dist_list
    #Need way to know expected number of parameters - should match output of param_transforms
    self.num_params = num_params
    #Define transformations on parameters (for instance, if predict log(var), take exp)
    if param_transforms is None:
      self.param_transforms = [lambda x, y: [x, tf.math.exp(0.5*y)]]*coordinate_dimensionality #Defaults to mean and log(var) of Normal distribution
    else:
      self.param_transforms = param_transforms
    #Also need to define transformations on mean functions
    #For some distributions, mean is constrained, so must transform output on (-inf, inf)
    if mean_transforms is None:
      self.mean_transforms = [tf.identity]*coordinate_dimensionality
    else:
      self.mean_transforms = mean_transforms
    #Define coordinate transformation
    self.coord_transform = coord_transform
    #And hidden dimension for all networks
    self.hidden_dim = hidden_dim

  def get_log_probs(self, coords, params, ref_coords):
    """Obtains log probabilities for this decoding model.
    Inputs:
            coords - coordinates (global cartesian) to assess probability of (n_batch, n_particles, n_dimensions)
            params - parameters that define probability distributions for coords (n_batch, n_particles, n_dimensions, n_params)
            ref_coords - reference coordinates to define local coordinate system for probs
    Outputs:
            log_probs - log probabilities of each PARTICLE, so the last dimension of coordinates has been summed over the individual probabilities
    """
    #Transform to local, transformed coordinates
    local_coords = self.coord_transform(coords - ref_coords)
    dist = create_dist(params, self.tfp_dist_list, self.param_transforms)
    log_probs = dist.log_prob(tf.unstack(local_coords, axis=-1))
    return log_probs

  def generate_sample(self, params):
    """Generates a sample from given probability distribution parameters.
    Inputs:
            params - parameters that define probability distributions (n_batch, n_particles, n_dimensions, n_params)
    Outputs:
            sample - sample drawn from probability distributions defined by params
    """
    #Create distribution and draw sample to return
    dist = create_dist(params, self.tfp_dist_list, self.param_transforms)
    sample = tf.stack(dist.sample(), axis=-1)
    return sample

  def select_from_data(self, ref, means, data, data_mask, return_local=False):
    """Based on provided means of probability distributions, selects data coordinates closest to them. Data is masked to only select eligible data (i.e., not already used).
    Inputs:
            ref - reference coordinate that defines local coordinate system of means (n_batch, 1, n_dims)
            means - distribution means for selecting closest data (n_batch, 1, n_dims)
            data - training data to pick from (n_batch, n_particles, n_dims)
            data_mask - array specifying which indices from data are eligible (n_batch, n_eligible_particles)
            return_local - (False) whether or not to return data coordinates in the local, transformed coordinate system or the global, cartesian coordinates of a simulation
    Outputs:
            data_coords - select data coordinates (n_batch, 1, n_dims)
            data_mask - new data mask excluding selected data
    """
    #First transform means since may be different than parameters for given distribution
    unstacked_means = tf.unstack(means, axis=-1)
    trans_means = []
    for k in range(len(unstacked_means)):
      trans_means.append(self.mean_transforms[k](unstacked_means[k]))
    trans_means = tf.stack(trans_means, axis=-1)
    #Convert from location (or mean) parameter in internal coords to global
    global_locs = ref + self.coord_transform(trans_means, reverse=True)
    #Select closest point to mean
    masked_data = tf.gather(data, data_mask, axis=1, batch_dims=1)
    data_coords, data_inds, data_dists = self.distance_mask(global_locs, masked_data, k_neighbors=1)
    data_coords = data_coords + global_locs #self.distance_mask makes local around first arg
    #Mask out this training data point in future searches
    #Do by removing indices that of used points so won't be gathered next time
    mask_shape = tf.shape(data_mask)
    all_inds = tf.tile(tf.expand_dims(tf.range(mask_shape[-1]), 0), (mask_shape[0], 1))
    mask = tf.not_equal(all_inds, data_inds)
    data_mask = tf.reshape(tf.boolean_mask(data_mask, mask), (mask_shape[0], -1))
    #When populating solvent around solutes, want to return in local, transformed coords
    #And make sure it's local to reference particle, not particle parameters we're adding!
    if return_local:
      data_coords = self.coord_transform(data_coords - ref)
    else:
      #For solvent around solvent, want global to save computation
      data_coords = data_coords
    return data_coords, data_mask

  def call(self):
    """Not implemented here as this is intended as a base class to be inherited
    """
    raise NotImplementedError()


class SoluteDecoder(ParticleDecoder):
  """Decodes from solute coordinates, reintroducing solvent in solute hydration shells. Inherits from ParticleDecoder to define all class-based utility functions and basic parameters.
  """
  def __init__(self,
               solute_shell_num,
               name='solute_decoder',
               **kwargs):
    """Generates SoluteDecoder class object, creating all necessary neural networks (SolvationNet objects)
    Inputs:
            solute_shell_num - number of particles in the solute solvation shell to generate
    Outputs:
            SoluteDecoder object
    """
    super(SoluteDecoder, self).__init__(name=name, **kwargs)
    #Define output shape for solute networks - number shell particles by dimensionality
    self.solute_out_shape = (solute_shell_num, self.coord_dim)
    #Create all the networks that will be needed
    self.base_solute_net = SolvationNet(self.solute_out_shape,
                                       out_event_dims=self.num_params,
                                       hidden_dim=self.hidden_dim,
                                       n_hidden=3,
                                       augment_input=False)
    self.solute_net = SolvationNet(self.solute_out_shape,
                                  out_event_dims=self.num_params,
                                  hidden_dim=self.hidden_dim,
                                  n_hidden=3,
                                  augment_input=True)

  def call(self, solute_coords, train_data=None):
    """Given input solute coordinates (and potentially solvent coordinate training data), generates solvent coordinates around solutes.
    Inputs:
            solute_coords - solute coordinates to use as "latent" or "CG" variables to decode into solvating solvent (n_batch, n_particles, n_dimensions)
            train_data - (None) training data with solvent coordinates (n_batch, n_particles, n_dimensions)
    Outputs:
            solvent_out - generated solvent coordinates or solvent coords selected from train_data that are closest to generated probability distribution means (n_batch, n_particles, n_dimensions)
            params_out - generated probability distribution parameters (n_batch, n_particles, n_dimensions, n_params)
            ref_out - each solvent coordinate was generated/had its probability assessed in local, transformed coordinate system with these coordinates defining the origin (n_batch, n_particles, n_dimensions)
            unused_data - training data that was not selected and is left-over (in this case, should be applying SolventDecoder next to finish decoding)
    """
    #Keep track of output solvent coordinates
    #(must do throughout because autoregressive)
    solvent_out = tf.reshape(tf.convert_to_tensor([]),
                             (tf.shape(solute_coords)[0], 0, self.coord_dim))
    #Also output parameters
    params_out = tf.reshape(tf.convert_to_tensor([]),
                            (tf.shape(solute_coords)[0], 0,
                             self.coord_dim, self.num_params))
    #But parameters in local, transformed coords, so also track reference coords for params
    ref_out = tf.reshape(tf.convert_to_tensor([]),
                         (tf.shape(solute_coords)[0], 0, self.coord_dim))
    #Need way to exclude some training data points and only expose those that are unused
    if train_data is not None:
      data_mask = tf.tile(tf.expand_dims(tf.range(train_data.shape[1], dtype=tf.int32), 0),
                          (train_data.shape[0], 1))
    else:
      data_mask = tf.ones((1, 1))

    #For each solute coordinate, loop over and add solvation shell
    for i in range(solute_coords.shape[1]):
      #Need below so autograph works because all these things change shape each loop
      tf.autograph.experimental.set_loop_options(
                shape_invariants=[(solvent_out, tf.TensorShape([None, None, None])),
                                  (params_out, tf.TensorShape([None, None, None, None])),
                                  (ref_out, tf.TensorShape([None, None, None])),
                                  (data_mask, tf.TensorShape([None, None]))])
      #Find nearest solute neighbors
      ref_coord = solute_coords[:, i:i+1, :] #[n_batch, n_particles, n_coordinate]
      #Including reference for distance mask, so indicate so handles properly
      k_solute_coords, k_solute_inds, k_solute_dists = self.distance_mask(ref_coord,
                                                                     solute_coords,
                                                        k_neighbors=self.k_solute_neighbors,
                                                        ref_included=True)
      #Transform coordinates
      k_solute_coords = self.coord_transform(k_solute_coords)
      #Change behavior if first solute
      if i == 0:
        #Predict initial parameters and shifts
        params, shifts = self.base_solute_net(k_solute_coords)
      else:
        #Again initial params and shifts, but now need closest solvent as well
        k_solvent_coords, k_solvent_inds, k_solvent_dists = self.distance_mask(ref_coord,
                                                                          solvent_out,
                                                         k_neighbors=self.k_solvent_neighbors)
        k_solvent_coords = self.coord_transform(k_solvent_coords)
        params, shifts = self.solute_net(k_solute_coords, extra_coords=k_solvent_coords)
      #On generation, sample from distribution, while on training, select closest particle
      if train_data is not None:
        #Select closest point of data to mean parameters as our "sample"
        sample, data_mask = self.select_from_data(ref_coord,
                                                  params[:, :1, :, 0] + shifts[:, :1, :, 0],
                                                  train_data,
                                                  data_mask,
                                                  return_local=True)
        #Set "sampled" coordinates to identified data point, padded with base param means
        sample = tf.concat([sample, params[:, 1:, :, 0]], axis=1)
      else:
        #Sample first particle in solvation shell
        sample = self.generate_sample(params + shifts)
      this_solvent_out = tf.identity(sample)

      #Loop over rest of solvation shell to make autoregressive
      for j in range(1, self.solute_net.out_shape[0]):
        tf.autograph.experimental.set_loop_options(
              shape_invariants=[(this_solvent_out, tf.TensorShape([None, None, None]))])
        if i == 0:
          shifts = self.base_solute_net(k_solute_coords, sampled_input=this_solvent_out)
        else:
          shifts = self.solute_net(k_solute_coords,
                                   extra_coords=k_solvent_coords,
                                   sampled_input=this_solvent_out)
        if train_data is not None:
          sample, data_mask = self.select_from_data(ref_coord,
                                              params[:, j:j+1, :, 0] + shifts[:, j:j+1, :, 0],
                                              train_data,
                                              data_mask,
                                              return_local=True)
          #Insert data point select ("sample") at jth position
          sample = tf.concat([params[:, :j, :, 0], sample, params[:, j+1:, :, 0]], axis=1)
        else:
          sample = self.generate_sample(params + shifts)
        #Leave items less than index j unchanged in this_solvent_out, update rest
        this_solvent_out = tf.concat([this_solvent_out[:, :j, :], sample[:, j:, :]], axis=1)

      #Before moving to next solute particle, update solvent coordinates
      #Make sure to transform out of local coordinate system
      solvent_out = tf.concat([solvent_out,
                         ref_coord + self.coord_transform(this_solvent_out, reverse=True)],
                              axis=1)
      params_out = tf.concat([params_out, params + shifts], axis=1)
      ref_out = tf.concat([ref_out,
                           tf.tile(ref_coord, (1, tf.shape(this_solvent_out)[1], 1))],
                          axis=1)

    if train_data is not None:
      #Will need to know which solvent were used and which weren't, so return unused as well
      unused_data = tf.gather(train_data, data_mask, axis=1, batch_dims=1)
    else:
      unused_data = None

    return solvent_out, params_out, ref_out, unused_data


class SolventDecoder(ParticleDecoder):
  """Decodes from solvent coordinates, reintroducing one new solvent for each one present in each neural network layer. Inherits from ParticleDecoder to define all class-based utility functions and basic parameters.
  """
  def __init__(self,
               num_solvent_nets,
               bulk_solvent=False,
               name='solvent_decoder',
               **kwargs):
    """Generates SolventDecoder class object, creating all necessary neural networks (SolvationNet objects)
    Inputs:
            num_solvent_nets - number of nets to apply to solvent (will double number with each net)
            bulk_solvent - (False) boolean for whether or not this applies to bulk solvent (so that solute coordinates should or should not be considered/expected)
    Outputs:
            SolventDecoder object
    """
    super(SolventDecoder, self).__init__(name=name, **kwargs)
    #Need to know if solutes are present (or if working with bulk solvent)
    self.bulk = bulk_solvent
    #Solvent nets to be handled as list because number may vary
    #Each network will double the number of solvent particles present
    #(by adding 1 new particle for each current particle)
    self.solvent_nets = []
    for i in range(num_solvent_nets):
      self.solvent_nets.append(SolvationNet((1, self.coord_dim),
                                           out_event_dims=self.num_params,
                                           hidden_dim=self.hidden_dim,
                                           n_hidden=3,
                                           augment_input=(not self.bulk)))

  def call(self, solvent_coords, solute_coords=None, train_data=None):
    """Given input solvent coordinates (and potentially solute coordinates or solvent training data), generates solvent coordinates around solvent.
    Inputs:
            solvent_coords - solvent coordinates to use as starting positions to add solvent around (n_batch, n_particles, n_dimensions)
            solute_coords - (None) if not bulk system (default) must add solute_coords to consider solute when placing new solvent (n_batch, n_particles, n_dimensions)
            train_data - (None) training data with solvent coordinates (n_batch, n_particles, n_dimensions)
    Outputs:
            solvent_out - generated solvent coordinates or solvent coords selected from train_data that are closest to generated probability distribution means (n_batch, n_particles, n_dimensions)
            params_out - generated probability distribution parameters (n_batch, n_particles, n_dimensions, n_params)
            ref_out - each solvent coordinate was generated/had its probability assessed in local, transformed coordinate system with these coordinates defining the origin (n_batch, n_particles, n_dimensions)
            unused_data - training data that was not selected and is left-over (in this case, should be applying another SolventDecoder to finish decoding)
    """
    batch_dim = tf.shape(solvent_coords)[0]
    #Keep track of output solvent coordinates
    #(must do throughout because autoregressive)
    solvent_out = tf.reshape(tf.convert_to_tensor([]),
                             (batch_dim, 0, self.coord_dim))
    #Also output parameters
    params_out = tf.reshape(tf.convert_to_tensor([]),
                            (batch_dim, 0, self.coord_dim, self.num_params))
    #But parameters in local, transformed coords, so also track reference coords for params
    ref_out = tf.reshape(tf.convert_to_tensor([]),
                         (batch_dim, 0, self.coord_dim))
    #Need way to exclude some training data points and only expose those that are unused
    if train_data is not None:
      data_mask = tf.tile(tf.expand_dims(tf.range(train_data.shape[1], dtype=tf.int32), 0),
                          (batch_dim, 1))
    else:
      data_mask = tf.ones((1, 1))

    #Solvation shells should already be placed, or may not be relevant
    #Loop through solvent networks, applying to each solvent particle
    #Good news is that for each solvent, just predict 1 particle position
    #So no need for a second autoregressive loop!
    #Will be autoregressive by virture of fact that keep adding to solvent_out
    curr_num_solv = solvent_coords.shape[1]
    solv_count = 0
    for solv_net in self.solvent_nets:
      curr_num_solv = curr_num_solv + solv_count
      solv_count = 0
      for i in range(curr_num_solv):
        #Need below so autograph works because all these things change shape each loop
        tf.autograph.experimental.set_loop_options(
                shape_invariants=[(solvent_out, tf.TensorShape([None, None, None])),
                                  (params_out, tf.TensorShape([None, None, None, None])),
                                  (ref_out, tf.TensorShape([None, None, None])),
                                  (data_mask, tf.TensorShape([None, None]))])
        #Concatenate the solvent input and output each time to make sure it's up to date
        #But note loop will only run over solvent input and any solvent added on last loop
        this_solvent = tf.concat([solvent_coords, solvent_out], axis=1)
        #Get nearby solutes and solvents
        ref_coord = this_solvent[:, i:i+1, :]
        if not self.bulk:
          k_solute_coords, k_solute_inds, k_solute_dists = self.distance_mask(ref_coord,
                                                                         solute_coords,
                                                          k_neighbors=self.k_solute_neighbors)
          k_solute_coords = self.coord_transform(k_solute_coords)
        else:
          k_solute_coords = None
        k_solvent_coords, k_solvent_inds, k_solvent_dists = self.distance_mask(ref_coord,
                                                                          this_solvent,
                                                         k_neighbors=self.k_solvent_neighbors,
                                                         ref_included=True)
        k_solvent_coords = self.coord_transform(k_solvent_coords)
        #Get parameters and shifts from network
        params, shifts = solv_net(k_solvent_coords, extra_coords=k_solute_coords)
        if train_data is not None:
          sample, data_mask = self.select_from_data(ref_coord,
                                                    params[..., 0] + shifts[..., 0],
                                                    train_data,
                                                    data_mask,
                                                    return_local=False)
        else:
          sample = self.generate_sample(params + shifts)
          sample = ref_coord + self.coord_transform(sample, reverse=True)
        #Add sample directly to solvent_out
        solvent_out = tf.concat([solvent_out, sample], axis=1)
        params_out = tf.concat([params_out, params + shifts], axis=1)
        ref_out = tf.concat([ref_out, ref_coord], axis=1)
        solv_count += 1

    if train_data is not None:
      #May need to know which solvent were used and which weren't, so return unused as well
      unused_data = tf.gather(train_data, data_mask, axis=1, batch_dims=1)
    else:
      unused_data = None

    return solvent_out, params_out, ref_out, unused_data


class SoluteSolventDecoder(ParticleDecoder):
  """Convenience class to join together solute and solvent decoders
  """
  def __init__(self,
               solute_shell_num,
               num_solvent_nets,
               name='decoder',
               **kwargs):
    """Generates SoluteSolventDecoder class object, by placing a SolventDecoder after a SoluteDecoder
    Inputs:
            solute_shell_num - number of particles in the solute solvation shell to generate
            num_solvent_nets - number of nets to apply to solvent (will double number with each net)
    Outputs:
            SoluteSolventDecoder object
    """
    super(SoluteSolventDecoder, self).__init__(name=name, **kwargs)
    self.solute_decoder = SoluteDecoder(solute_shell_num, **kwargs)
    self.solvent_decoder = SolventDecoder(num_solvent_nets, bulk_solvent=False, **kwargs)

  def call(self, solute_coords, train_data=None):
    """Given input solute coordinates (and potentially solvent coordinate training data), generates solvent coordinates around solutes.
    Inputs:
            solute_coords - solute coordinates to use as "latent" or "CG" variables to decode into solvating solvent (n_batch, n_particles, n_dimensions)
            train_data - (None) training data with solvent coordinates (n_batch, n_particles, n_dimensions)
    Outputs:
            solvent_out - generated solvent coordinates or solvent coords selected from train_data that are closest to generated probability distribution means (n_batch, n_particles, n_dimensions)
            params_out - generated probability distribution parameters (n_batch, n_particles, n_dimensions, n_params)
            ref_out - each solvent coordinate was generated/had its probability assessed in local, transformed coordinate system with these coordinates defining the origin (n_batch, n_particles, n_dimensions)
            unused_data - training data that was not selected and is left-over (in this case, should be applying SolventDecoder next to finish decoding)
    """
    sample_out1, params_out1, ref_out1, unused_data = self.solute_decoder(solute_coords, train_data=train_data)
    sample_out2, params_out2, ref_out2, unused_data = self.solvent_decoder(sample_out1, solute_coords=solute_coords, train_data=unused_data)

    solvent_out = tf.concat([sample_out1, sample_out2], axis=1)
    params_out = tf.concat([params_out1, params_out2], axis=1)
    ref_out = tf.concat([ref_out1, ref_out2], axis=1)

    return solvent_out, params_out, ref_out, unused_data


class DecimationEncoder(tf.keras.layers.Layer):
  """Encoding with fixed, deterministic decimation mapping.
  """

  def __init__(self,
               cg_particle_mask,
               identical_randomize=False,
               name='encoder',
               **kwargs):
    """Generates DecimationEncoder layer that selects out CG particles
    Inputs:
            cg_particle_mask - boolean mask to specify which particles are CG or not
            identical_randomize - (False) if CG sites are identical, may want to randomize labels to encourage better generalization
    Outputs:
            DecimationEncoder object
    """
    super(DecimationEncoder, self).__init__(name=name, **kwargs)
    self.cg_mask = cg_particle_mask
    self.randomize = identical_randomize
    self.num_cg = int(np.sum(self.cg_mask))
    self.num_non_cg = int(len(self.cg_mask) - self.num_cg)

  def call(self, input_coords):
    """Selects out CG particle coordinates and returns from full input coordinate set
    Inputs:
            input_coords - coordinates of all particles in full system (n_batch, n_particles, n_dimensions)
    Outputs:
            cg_coords - coordinates of just the CG particles
            non_cg_coords - coordinates of non-CG particles
    """
    cg_coords = tf.boolean_mask(input_coords, self.cg_mask, axis=1)
    non_cg_coords = tf.boolean_mask(input_coords, tf.math.logical_not(self.cg_mask), axis=1)
    if self.randomize:
      #cg_coords = tf.transpose(tf.random.shuffle(tf.transpose(cg_coords, perm=[1, 0, 2])),
      #                         perm=[1, 0, 2])
      #Above is fast, but all batch samples have particles shuffled the same way
      cg_coords = tf.map_fn(tf.random.shuffle, cg_coords)
    #Reshape to ensure that the output shape is well-defined based on input
    cg_coords = tf.reshape(cg_coords,
                         (tf.shape(input_coords)[0], self.num_cg, tf.shape(input_coords)[-1]))
    non_cg_coords = tf.reshape(non_cg_coords,
                     (tf.shape(input_coords)[0], self.num_non_cg, tf.shape(input_coords)[-1]))
    return cg_coords, non_cg_coords


class PriorFlowSolventVAE(tf.keras.Model):
  """VAE with fixed, decimation encoding and prior flow to learn the CG potential, then with a particle decoding. This allows for the construction of a typical implicit "solvent" CG model while also learning a back-mapping to re-introduce solvent. Intended for when have bulk solvent and want to "CG" through decimation.
  """

  def __init__(self,
               data_shape,
               num_halves,
               decoder_kwargs=None,
               flow_type='rqs',
               flow_net_params=None,
               beta=1.0,
               name='priorflow_vae',
               **kwargs):
    """Creates VAE for decimation encoding and decoding of bulk solvent
    Inputs:
            data_shape - shape of the full data for the system (n_particles, n_dimensions)
            num_halves - number of times the number of particles should be reduced by half to define the "coarse-grained" particles
            decoder_kwargs - (None) dictionary of keyword arguments for the decoder
            flow_type - ('rqs') can be 'affine' for standard RealNVP or 'rqs' for RealNVP structure with neural splines for the transformation instead
            flow_net_params = (None) dictionary of keyword arguments for flow
            beta - (1.0) weight on the KL divergence term; here it's not actually a KL divergence term, it just allows for turning on (1) or off (0) the training of the flow in the CG space
    Outputs:
            PriorFlowSolventVAE class instance
    """
    super(PriorFlowSolventVAE, self).__init__(name=name, **kwargs)
    self.beta = beta #Beta for switching on/off learning prior, similar to Beta-VAE
    self.data_shape = data_shape #Should specify (n_particles, n_dimensions) in full system
    #To define encoding, specify number of times to halve the number of bulk solvent particles
    self.num_cg = data_shape[0] / (2*num_halves)
    self.num_latent = self.num_cg*self.data_shape[1]
    #And use this to uniformly select out indices of solvent
    cg_inds = np.arange(0, self.data_shape[0], self.data_shape[0]/self.num_cg, dtype=int)
    cg_mask = np.zeros(self.data_shape[0], dtype=bool)
    cg_mask[cg_inds] = True
    self.encoder = DecimationEncoder(cg_mask, identical_randomize=True)
    #And next define decoder
    if decoder_kwargs is None:
      #By default set box as if all coordinates normalized to be between -1 and 1
      decoder_kwargs = {'box_lengths':np.array([2.0, 2.0, 2.0])}
    self.decoder = SolventDecoder(num_halves, bulk_solvent=True, **decoder_kwargs)
    if flow_type == 'affine':
      if flow_net_params is None:
        flow_net_params = {'num_hidden':2, 'hidden_dim':2*self.num_cg,
                           'nvp_split':True, 'activation':tf.nn.relu}
      self.flow = NormFlowRealNVP(self.num_latent,
                                  kernel_initializer='truncated_normal',
                                  flow_net_params=flow_net_params,
                                  num_blocks=4)
    elif flow_type == 'rqs':
      if flow_net_params is None:
        #For bin range with decimation mapping and periodic simulation box,
        #should really use the box dimensions as the bin range
        #If different in different dimensions, normalize coordinates so between -1 and 1
        #And make sure all wrapped into the box!
        flow_net_params = {'bin_range':[-1.0, 1.0], 'num_bins':32,
                           'hidden_dim':2*self.num_cg}
      self.flow = NormFlowRQSplineRealNVP(self.num_latent,
                                          kernel_initializer='truncated_normal',
                                          rqs_params=flow_net_params)
    else:
      raise ValueError("Specified flow type \'%s\' is not known. Should be \'affine\' or \'rqs\'")
    #If have encoder that doesn't go to 1D, need to flatten encoding for flow
    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs, training=False):
    cg, non_cg = self.encoder(inputs)
    z = self.flatten(cg)
    #In this model, no KL divergence, but still want to maximize likelihood of P(z)
    #Here we define P(z) as a flow over a standard normal prior
    #So pass z through reverse flow to estimate likelihood
    #Should be able to do this AFTER training if like, but testing that idea out
    #If do after, MUST do really well before actually using model in MC simulations
    if self.beta != 0.0:
      #If regularization is zero, save time on the calculation
      z_prior, logdet = self.flow(z, reverse=True)
      #Estimate (negative) log likelihood of the prior
      logp_z = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(z_prior)
                                                + tf.math.log(2.0*np.pi),
                                                axis=1))
      #And SUBTRACT the average log determinant for the flow transformation
      logp_z -= tf.reduce_mean(logdet)
    else:
      logp_z = 0.0
    #With flow only on prior, z passes directly through
    if training:
      #Only pass non_cg coords as training data to decoder since not reconstructing cg parts
      recon_info = self.decoder(cg, train_data=non_cg)
    else:
      recon_info = self.decoder(cg)
    reg_loss = self.beta*logp_z
    #Add losses within here - keeps code cleaner and less confusing
    self.add_loss(reg_loss)
    self.add_metric(tf.reduce_mean(logp_z), name='logp_z', aggregation='mean')
    self.add_metric(tf.reduce_mean(reg_loss), name='regularizer_loss', aggregation='mean')
    #And do reconstruction loss, too
    #With autoregessive models, the proabilities are defined within the decoder itself
    #So doesn't make sense to have external loss function, really
    recon_loss = tf.reduce_mean(tf.reduce_sum(self.decoder.get_log_probs(*recon_info[:-1]),
                                              axis=1))
    self.add_loss(recon_loss)
    self.add_metric(tf.reduce_mean(recon_loss), name='recon_loss', aggregation='mean')
    return recon_info


