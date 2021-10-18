
import sys, os
import gc
import time

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

from libVAE import dataloaders, losses, vae, mc_moves, mc_moves_LG

import simtk.unit as unit


#Define function to save trajectory if we want to
#Same as in simLG.sim_2D
def createDataFile(fileName, L, biasRange=None):
  """Create and return a netcdf Dataset object to hold simulation data.
     Handles to important variables within the dataset are also returned."""

  outDat = Dataset(fileName, "w", format="NETCDF4") 
  outDat.history = "Created " + time.ctime(time.time())
  outDat.createDimension("steps", None)
  outDat.createDimension("x", L)
  outDat.createDimension("y", L)
  if biasRange is None:
    outDat.createDimension("N", L*L+1)
  else:
    outDat.createDimension("N", biasRange[1]-biasRange[0]+1)
  steps = outDat.createVariable("steps", "u8", ("steps",))
  steps.units = "MC steps"
  x = outDat.createVariable("x", "i4", ("x",))
  x.units = "lattice unit"
  x[:] = np.arange(L, dtype=int)
  y = outDat.createVariable("y", "i4", ("y",))
  y.units = "lattice unit"
  y[:] = np.arange(L, dtype=int)
  U = outDat.createVariable("U", "f8", ("steps",))
  U.units = "energy"
  config = outDat.createVariable("config", "u1", ("steps", "x", "y",))
  config.units = "particle positions on lattice"
  N = outDat.createVariable("N", "i4", ("N",))
  N.units = "number particles"
  if biasRange is None:
    N[:] = np.arange(L*L+1, dtype=int)
  else:
    N[:] = np.arange(biasRange[0], biasRange[1]+1, dtype=int)
  biasVals = outDat.createVariable("bias", "f8", ("steps", "N",))
  biasVals.units = "dimensionless energy" 

  return outDat, steps, U, config, biasVals


#For transforming coordinates for molecular systems...
#Works for any molecular system with dihedrals or sine-cosine pairs last (default)
#Just specifiy the total number of DOFs for the molecule
def sincos(x, totDOFs):
  torsion_sin = np.sin(x[:, -(totDOFs//3 - 3):])
  torsion_cos = np.cos(x[:, -(totDOFs//3 - 3):])
  out_x = np.concatenate([x[:, :-(totDOFs//3 - 3)], torsion_sin, torsion_cos], axis=1)
  return out_x


#Unlike sincos, unsincos just does one config at a time to work with bat_analysis
def unsincos(x, totDOFs):
  sin_vals = x[-2*(totDOFs//3 - 3):-(totDOFs//3 - 3)]
  cos_vals = x[-(totDOFs//3 - 3):]
  r_vals = np.sqrt(sin_vals**2 + cos_vals**2)
  torsion_vals = np.arctan2(sin_vals/r_vals, cos_vals/r_vals)
  out_x = np.concatenate([x[:-2*(totDOFs//3 - 3)], torsion_vals])
  return out_x


#Get system type
system_type = sys.argv[1]

#Get VAE parameters to load
weights_file = sys.argv[2]

#Collect data files to work with
dat_files = sys.argv[3:]

#Set up VAE model based on system type and weights_file
latent_dim = weights_file.split('_l')
#If don't have '_lX' to split on, assume isn't specified, so cgmodel and has 1D latent
if len(latent_dim) == 1:
  latent_dim = 1
else:
  latent_dim = int(latent_dim[-1].split('/')[0])

if system_type == 'lg':
  print("Setting up VAE for lattice gas with latent dimension %i"%latent_dim)
  vaeModel = vae.PriorFlowVAE((28, 28, 1), latent_dim, autoregress=True)
  energy_func = losses.latticeGasHamiltonian
  energy_params = {'mu':-2.0, 'eps':-1.0}
  sampler_params = {'activation':None,
                    'sampler_func':losses.binary_sampler,
                    'logp_func':losses.bernoulli_loss}
  beta = 1.77
  data = dataloaders.raw_image_data(dat_files)
  #Should add command line option for saving to file, but instead adding flag here
  write_traj = True

#If do add other systems, must set up energy functions for them as well!
elif system_type == 'dimer':
  print("Setting up VAE for 2D particle dimer with latent dimension %i"%latent_dim)
  #vaeModel = vae.PriorFlowVAE((76,), latent_dim, autoregress=True,
  #                            include_vars=True, n_auto_group=2,
  #                            e_hidden_dim=300,
  #                            d_hidden_dim=300)
  vaeModel = vae.FullFlowVAE((76,), latent_dim, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)

  trajdict = np.load(dat_files[0], allow_pickle=True)
  data = np.vstack([trajdict['traj_open_hungarian'], trajdict['traj_closed_hungarian']])
  if 'centerwhite' in weights_file:
    shift = np.mean(data, axis=0)
  else:
    shift = 0.0
  data -= shift
  if 'centerwhite' in weights_file:
    scale = np.std(data, axis=0, ddof=1)
  else:
    scale = 1.0
  data /= scale

  def cw_energy(conf):
    return losses.dimerHamiltonian(conf*scale + shift).numpy()

  energy_func = cw_energy
  energy_params = {}
  sampler_params = {'activation':None,
                    'sampler_func':None, #vaeModel.decoder.create_sample,
                    'logp_func':losses.AutoregressiveLoss(vaeModel.decoder,
                                                   reduction=tf.keras.losses.Reduction.NONE)}
  beta = 1.00
  write_traj = False

elif system_type == 'ala':
  print("Setting up VAE for dialanine with latent dimension %i"%latent_dim)
  periodic_inds = list(range(-19, 0))
  if 'periodic' in weights_file:
    vaeModel = vae.PriorFlowVAE((48,), latent_dim, autoregress=True,
                                include_vars=True, n_auto_group=1,
                                periodic_dof_inds=periodic_inds,
                                e_hidden_dim=300,
                                d_hidden_dim=300)
  elif 'sincos' in weights_file:
    vaeModel = vae.PriorFlowVAE((67,), latent_dim, autoregress=True,
                                include_vars=True, n_auto_group=1,
                                e_hidden_dim=300,
                                d_hidden_dim=300)

  import MDAnalysis as mda
  from MDAnalysis.analysis import bat
  prmtopFile = os.path.expanduser('~/CG_MC/Ala_Dipeptide/alanine-dipeptide.prmtop')
  strucFile = os.path.expanduser('~/CG_MC/Ala_Dipeptide/alanine-dipeptide.pdb')
  uni = mda.Universe(prmtopFile, strucFile)
  bat_analysis = bat.BAT(uni.select_atoms('all'))

  #Define mask for bonds and rigid rotation/translation (constrained DOFs)
  totDOFs = 66
  bond_inds = list(range(6))
  bond_inds = bond_inds + [9, 11, 12, 13, 15, 17, 18, 19, 21, 24, 25, 26]
  bond_mask = np.ones(totDOFs, dtype=bool)
  bond_mask[bond_inds] = False

  def transform(x):
    #Turn sine-cosine pairs back into dihedrals
    if 'sincos' in weights_file:
      x = unsincos(x, totDOFs)
    #Unmask
    out_x = np.zeros(totDOFs)
    out_x[bond_mask] = x
    out_x[np.invert(bond_mask)] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, #Rigid rotation and translation
                                            1.01, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.01, 1.09, 1.09, 1.09])
                                            #Bond lengths for masked bonds with hydrogens - NOT all same
                                            #1.01 is for N-H and 1.09 is for C-H
    #Convert from BAT to XYZ
    out_x = bat_analysis.Cartesian(out_x)
    return out_x

  energy_func = losses.dialanineHamiltonian(transform_func=transform)
  energy_params = {}
  sampler_params = {'activation':None,
                    'sampler_func':vaeModel.decoder.create_sample,
                    'logp_func':losses.AutoregressiveLoss(vaeModel.decoder,
                                                   reduction=tf.keras.losses.Reduction.NONE)}
  beta = 1.0 / (300.0*unit.kelvin * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilojoules_per_mole)

  data = np.load(dat_files[0])
  data = data[:, bond_mask]
  if 'sincos' in weights_file:
    data = sincos(data, totDOFs)
  write_traj = False

elif 'poly' in system_type:
  print("Setting up VAE for polymer with latent dimension %i"%latent_dim)
  periodic_inds = list(range(-17, 0))
  if 'periodic' in weights_file:
    vaeModel = vae.PriorFlowVAE((35,), latent_dim, autoregress=True,
                                include_vars=True, n_auto_group=1,
                                periodic_dof_inds=periodic_inds,
                                e_hidden_dim=300,
                                d_hidden_dim=300)
  elif 'sincos' in weights_file:
    vaeModel = vae.PriorFlowVAE((52,), latent_dim, autoregress=True,
                                include_vars=True, n_auto_group=1,
                                e_hidden_dim=300,
                                d_hidden_dim=300)

  import MDAnalysis as mda
  from MDAnalysis.analysis import bat
  if 'soft' in system_type:
    topFile = os.path.expanduser('~/CG_MC/Polymer/C20_soft/c20_soft.top')
    prmtopFile = os.path.expanduser('~/CG_MC/Polymer/C20/c20.prmtop')
    strucFile = os.path.expanduser('~/CG_MC/Polymer/C20_soft/c20_soft.pdb')
  else:
    topFile = os.path.expanduser('~/CG_MC/Polymer/C20/c20.top')
    prmtopFile = os.path.expanduser('~/CG_MC/Polymer/C20/c20.prmtop')
    strucFile = os.path.expanduser('~/CG_MC/Polymer/C20/c20.pdb')
  uni = mda.Universe(prmtopFile, strucFile)
  bat_analysis = bat.BAT(uni.select_atoms('all'))

  #Define mask for bonds and rigid rotation/translation (constrained DOFs)
  totDOFs = 60
  bond_inds = list(range(8))
  for b in range(9, 9 + totDOFs//3 - 3):
    bond_inds.append(b)
  bond_mask = np.ones(totDOFs, dtype=bool)
  bond_mask[bond_inds] = False

  def transform(x):
    #Turn sine-cosine pairs back into dihedrals
    if 'sincos' in weights_file:
      x = unsincos(x, totDOFs)
    #Unmask
    out_x = np.zeros(totDOFs)
    out_x[bond_mask] = x
    out_x[np.invert(bond_mask)] = 1.54
    out_x[:6] = np.zeros(6)
    #Convert from BAT to XYZ
    out_x = bat_analysis.Cartesian(out_x)
    return out_x

  energy_func = losses.polymerHamiltonian(topFile, strucFile, transform_func=transform)
  energy_params = {}
  sampler_params = {'activation':None,
                    'sampler_func':vaeModel.decoder.create_sample,
                    'logp_func':losses.AutoregressiveLoss(vaeModel.decoder,
                                                   reduction=tf.keras.losses.Reduction.NONE)}
  beta = 1.0 / (300.0*unit.kelvin * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilojoules_per_mole)
  
  data = np.load(dat_files[0])
  data = data[:, bond_mask]
  if 'sincos' in weights_file:
    data = sincos(data, totDOFs)
  write_traj = False
    
else:
  print("system type %s not recognized"%(system_type))
  sys.exit(2)

#Can write trajectory, but for now only with lattice gas
if write_traj:
  outDat, steps_nc, U_nc, config_nc, biasVals_nc = createDataFile('LG_2D_traj.nc', 28)
  write_freq = 5

#Load weights now
vaeModel.load_weights(weights_file)

print("Loaded weights from file: %s"%weights_file)
print("Loaded data from files: ", dat_files)

#Create method for drawing new z configurations
zDraw = mc_moves.zDrawFunc_wrap_VAE(data, vaeModel)

#Select random configurations as starting points for batch of MC simulations
num_parallel = 1000
rand_inds = np.random.choice(data.shape[0], size=num_parallel, replace=False)
curr_config = data[rand_inds]
curr_U = energy_func(curr_config, **energy_params)
if tf.is_tensor(curr_U):
  curr_U = curr_U.numpy()

#Allow for multiple types of MC moves
move_types = [mc_moves.moveVAE, #mc_moves.moveVAEbiased
              mc_moves_LG.moveTranslate,
              mc_moves_LG.moveDeleteMulti,
              mc_moves_LG.moveInsertMulti]
move_probs = [1.0, 0.0, 0.0, 0.0]

#Set up statistics
num_steps = 1000
move_counts = np.zeros((num_parallel, len(move_types)))
num_acc = np.zeros((num_parallel, len(move_types)))
if move_probs[0] > 0.0:
  mc_stats = np.zeros((num_parallel, num_steps, 8)) #8 if unbiased, 10 if biased
U_traj = np.zeros((num_parallel, num_steps+1))
U_traj[:, 0] = curr_U

if system_type == 'lg':
  N_traj = np.zeros((num_parallel, num_steps+1))
  N_traj[:, 0] = np.sum(curr_config, axis=(1, 2, 3))

if write_traj:
  steps_nc[0:num_parallel] = np.arange(num_parallel)
  U_nc[0:num_parallel] = curr_U
  config_nc[0:num_parallel, :, :] = curr_config[..., 0]
  #For next write, increase range to write over by num_parallel
  start_ind = num_parallel
  end_ind = 2*num_parallel

for i in range(num_steps):
    print('Step %i'%i)
    #Pick move type
    m = np.random.choice(np.arange(len(move_types)), p=move_probs)
    print('\tMove type %i'%m)
    move_counts[:, m] += 1
    if m == 0:
        move_info = move_types[m](curr_config, curr_U,
                                  vaeModel, beta, energy_func, zDraw,
                                  energyParams=energy_params, samplerParams=sampler_params,
                                  zDrawType='std_normal', verbose=True)
        mc_stats[:, i, :] = np.array(move_info[-1]).T
    else:
        move_info = move_types[m](curr_config, curr_U, beta)
    rand_logP = np.log(np.random.random(num_parallel))
    to_acc = (move_info[0]  > rand_logP)
    num_acc[to_acc, m] += 1.0
    curr_config[to_acc] = move_info[1][to_acc]
    curr_U[to_acc] = move_info[2][to_acc]
    U_traj[:, i+1] = curr_U

    if system_type == 'lg':
      N_traj[:, i+1] = np.sum(curr_config, axis=(1, 2, 3))

    if write_traj:
      if (i+1)%write_freq == 0:
        steps_nc[start_ind:end_ind] = np.arange(start_ind, end_ind)
        U_nc[start_ind:end_ind] = curr_U
        config_nc[start_ind:end_ind, :, :] = curr_config[..., 0]
        start_ind += num_parallel
        end_ind += num_parallel

    #tf.keras.backend.clear_session()
    gc.collect()

print("Acceptance rates: ")
for i in range(len(move_types)):
    print("Move type %i"%i, (num_acc[:, i]/move_counts[:, i]))
    print("Total: %f"%(np.sum(num_acc[:, i])/(np.sum(move_counts[:, i]))))
print("Move statistics: ")
print(np.sum(move_counts, axis=0))

if move_probs[0] > 0.0:
  np.save('mc_stats', mc_stats)

np.save('U', U_traj)

if system_type == 'lg':
  np.save('N', N_traj)

if write_traj:
  outDat.close()

