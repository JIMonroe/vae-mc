
import sys

import numpy as np
import tensorflow as tf

from libVAE import dataloaders, losses, vae, mc_moves


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
rand_inds = np.random.choice(data.shape[0], size=10, replace=False)
curr_config = data[rand_inds]
curr_U = losses.latticeGasHamiltonian(curr_config, **energy_params).numpy()

#Set up statistics
num_steps = 20
num_acc = np.zeros(curr_config.shape[0])
mc_stats = np.zeros((curr_config.shape[0], num_steps, 8))
N_traj = np.zeros((curr_config.shape[0], num_steps+1))
U_traj = np.zeros((curr_config.shape[0], num_steps+1))

N_traj[:, 0] = np.sum(curr_config, axis=(1, 2, 3))
U_traj[:, 0] = curr_U

for i in range(num_steps):
    print('Step %i'%i)
    move_info = mc_moves.moveVAE(curr_config, curr_U,
                             vaeModel, 1.77, energy_func, zDraw,
                             energyParams=energy_params, samplerParams=sampler_params,
                             zDrawType='std_normal', verbose=True)
    rand_logP = np.log(np.random.random(curr_config.shape[0]))
    mc_stats[:, i, :] = np.array(move_info[-1]).T
    to_acc = (move_info[0]  > rand_logP)
    num_acc[to_acc.numpy()] += 1.0
    curr_config[to_acc.numpy()] = move_info[1][to_acc]
    curr_U[to_acc.numpy()] = move_info[2][to_acc].numpy()
    N_traj[:, i+1] = np.sum(curr_config, axis=(1, 2, 3))
    U_traj[:, i+1] = curr_U

print("Acceptance rate: ", (num_acc/num_steps))
print("Total: %f"%(np.sum(num_acc)/(num_steps*curr_config.shape[0])))

np.save('mc_stats', mc_stats)
np.save('N', N_traj)
np.save('U', U_traj)


