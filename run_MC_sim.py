
import sys
import time

import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

from libVAE import dataloaders, losses, vae, mc_moves, mc_moves_LG


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


#Should add command line option for saving to file, but instead adding flag here
write_traj = True

if write_traj:
  outDat, steps_nc, U_nc, config_nc, biasVals_nc = createDataFile('LG_2D_traj.nc', 28)

#Get system type
system_type = sys.argv[1]

#Get VAE parameters to load
weights_file = sys.argv[2]

#Set up VAE model based on system type and weights_file
latent_dim = int(weights_file.split('_l')[-1].split('/')[0])
if system_type == 'lg':
  print("Setting up VAE for lattice gas with latent dimension %i"%latent_dim)
  vaeModel = vae.PriorFlowVAE((28, 28, 1), latent_dim, autoregress=True)
  energy_func = losses.latticeGasHamiltonian
  energy_params = {'mu':-2.0, 'eps':-1.0}
  sampler_params = {'activation':None,
                    'sampler_func':losses.binary_sampler,
                    'logp_func':losses.bernoulli_loss}
  beta = 1.77
#If do add other systems, must set up energy functions for them as well!
elif system_type == 'dimer':
  print("Setting up VAe for 2D particle dimer with latent dimension %i"%latent_dim)
  vaeModel = vae.PriorFlowVAE((76,), latent_dim, autoregress=True,
                              include_vars=True, n_auto_group=2)
  energy_func = losses.dimerHamiltonian
  energy_params = {}
  sampler_params = {'activation':None,
                    'sampler_func':losses.gaussian_sampler,
                    'logp_func':losses.diag_gaussian_loss}
  beta = 1.00
else:
  print("system type %s not recognized - can only be \'lg\' or \'dimer\'"%(system_type))
  sys.exit(2)

#Load weights now
vaeModel.load_weights(weights_file)

#Collect data files to work with
dat_files = sys.argv[3:]
data = dataloaders.raw_image_data(dat_files)

print("Loaded weights from file: %s"%weights_file)
print("Loaded data from files: ", dat_files)

#Create method for drawing new z configurations
zDraw = mc_moves.zDrawFunc_wrap_VAE(data, vaeModel)

#Select random configurations as starting points for batch of MC simulations
num_parallel = 1000
rand_inds = np.random.choice(data.shape[0], size=num_parallel, replace=False)
curr_config = data[rand_inds]
curr_U = losses.latticeGasHamiltonian(curr_config, **energy_params).numpy()

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
mc_stats = np.zeros((num_parallel, num_steps, 8)) #8 if unbiased, 10 if biased
N_traj = np.zeros((num_parallel, num_steps+1))
U_traj = np.zeros((num_parallel, num_steps+1))

N_traj[:, 0] = np.sum(curr_config, axis=(1, 2, 3))
U_traj[:, 0] = curr_U

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
    N_traj[:, i+1] = np.sum(curr_config, axis=(1, 2, 3))
    U_traj[:, i+1] = curr_U

    if write_traj:
      if (i+1)%5 == 0:
        steps_nc[start_ind:end_ind] = np.arange(start_ind, end_ind)
        U_nc[start_ind:end_ind] = curr_U
        config_nc[start_ind:end_ind, :, :] = curr_config[..., 0]
        start_ind += num_parallel
        end_ind += num_parallel

print("Acceptance rates: ")
for i in range(len(move_types)):
    print("Move type %i"%i, (num_acc[:, i]/move_counts[:, i]))
    print("Total: %f"%(np.sum(num_acc[:, i])/(np.sum(move_counts[:, i]))))
print("Move statistics: ")
print(np.sum(move_counts, axis=0))

np.save('mc_stats', mc_stats)
np.save('N', N_traj)
np.save('U', U_traj)

if write_traj:
  outDat.close()

