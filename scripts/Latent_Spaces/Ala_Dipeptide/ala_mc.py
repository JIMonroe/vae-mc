import sys, os
import gc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from libVAE import vae, losses, mc_moves
import MDAnalysis as mda
from libVAE.coord_transforms import BAT_tf, transform_ala_tf
import simtk.unit as unit

#Try to set GPU to allow multiple processes/trainings
#Assumes just have one GPU, so still set CUDA_VISIBLE_DEVICES
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

#Need gaussian sampling function that sets log-variances to 0 by default
#This allows decoders with no prediction of the variance
#And allows us to not modify code within mc_moves.py (or in losses.py)
def gaussian_sampler(mean, logvar=None):
    if logvar is None:
        logvar = tf.zeros_like(mean)
    return tf.add(mean, tf.exp(logvar / 2.0) * tf.random.normal(tf.shape(mean), 0, 1))


def diag_gaussian_loss(true_vals,
                       recon_info):
    if isinstance(recon_info, (tuple, list)):
        recon_means = recon_info[0]
        recon_log_var = recon_info[1]
    else:
        recon_means = recon_info
        recon_log_var = tf.zeros_like(recon_means)
    #Negative logP is represented below
    mse_term = 0.5*tf.square(true_vals - recon_means)*tf.exp(-recon_log_var)
    reg_term = 0.5*recon_log_var
    #norm_term = 0.5*tf.math.log(2.0*np.pi)
    sum_terms = tf.reduce_sum(mse_term + reg_term, # + norm_term,
                              axis=np.arange(1, len(true_vals.shape)))
    loss = sum_terms #Per sample loss
    return loss


#Define a function to run an MC simulation
def run_mc_sim(data, potential, model,
               draw_type='std_normal',
               num_steps=1000, write_freq=10,
               num_parallel=1000):

    #Grab random starting configurations
    rand_inds = np.random.choice(data.shape[0], size=num_parallel, replace=False)
    curr_config = data[rand_inds]
    curr_U = potential(curr_config)

    #Track acceptance and move statistics
    num_acc = np.zeros(num_parallel)
    mc_stats = np.zeros((num_parallel, num_steps, 8))

    #Record trajectory and potential energy
    traj = np.zeros((num_steps//write_freq + 1, num_parallel)+data.shape[1:], dtype='float32')
    traj_pot = np.zeros((num_steps//write_freq + 1, num_parallel), dtype='float32')
    traj[0, ...] = curr_config
    traj_pot[0, ...] = curr_U

    #Define how sampling of x performed
    if model.autoregress:
        sampler_params = {'activation':None,
                          'sampler_func':None, #Shouldn't actually need to set
                          'logp_func':losses.AutoregressiveLoss(model.decoder,
                                                                reduction=tf.keras.losses.Reduction.NONE)}
    else:
        sampler_params = {'activation':None,
                          'sampler_func':gaussian_sampler,
                          'logp_func':diag_gaussian_loss}

    #Define object for sampling latent space
    zDraw = mc_moves.zDrawFunc_wrap_VAE(data, model)

    thermo_beta = 1.0 / (300.0*unit.kelvin * unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilojoules_per_mole)

    for i in range(num_steps):
        move_info =  mc_moves.moveVAE(curr_config, curr_U,
                                      model, thermo_beta, potential, zDraw,
                                      energyParams={}, samplerParams=sampler_params,
                                      zDrawType=draw_type, verbose=True)
        rand_logP = np.log(np.random.random(num_parallel))
        to_acc = (move_info[0] > rand_logP)
        num_acc[to_acc] += 1.0
        mc_stats[:, i, :] = np.array(move_info[-1]).T
        curr_config[to_acc] = move_info[1][to_acc]
        curr_U[to_acc] = move_info[2][to_acc]

        if i%write_freq == 0:
            traj[i//write_freq + 1, ...] = curr_config
            traj_pot[i//write_freq + 1, ...] = curr_U

    print('\tTotal avg acc rate: %f'%(np.sum(num_acc)/(num_steps*num_parallel)))

    return curr_config, curr_U, traj, traj_pot, num_acc, mc_stats


def main(data_file, model_type, base_dir='.'):
    #Define key VAE parameters and name
    params_vae = {'e_hidden_dim':300,
                  'f_hidden_dim':20,
                  'd_hidden_dim':300,}
    vae_class = vae.PriorFlowVAE
    beta_vals = [0.0, 1.0, 2.0, 5.0, 10.0]
    latent_sizes = [1, 2, 4]

    #Load in data - MUST be loadable through np.load, so any preprocessing should be applied
    raw_data = np.load(data_file).astype('float32')

    #System-specific info
    totDOFs = raw_data.shape[1]
    bond_inds = list(range(6))
    bond_inds = bond_inds + [9, 11, 12, 13, 15, 17, 18, 19, 21, 24, 25, 26]
    bond_mask = np.ones(totDOFs, dtype=bool)
    bond_mask[bond_inds] = False

    #Specify periodic DOFs for specific system - CHANGE BASED ON SYSTEM!
    if 'BAT' in data_file:
        params_vae['periodic_dof_inds'] = list(range(-19, 0))
        raw_data = raw_data[:, bond_mask]

    #Define model and parameters
    if model_type == 'AE':
        params_vae['sample_latent'] = False
        params_vae['include_vars'] = False
        params_vae['autoregress'] = False
        beta_vals = [0.0]
    elif model_type == 'noAuto_noSigma':
        params_vae['sample_latent'] = True
        params_vae['include_vars'] = False
        params_vae['autoregress'] = False
    elif model_type == 'noAuto':
        params_vae['sample_latent'] = True
        params_vae['include_vars'] = True
        params_vae['autoregress'] = False
    elif model_type == 'auto':
        params_vae['sample_latent'] = True
        params_vae['include_vars'] = True
        params_vae['autoregress'] = True
    elif model_type == 'flow':
        vae_class = vae.FullFlowVAE
    else:
        raise ValueError('Specified model type of %s not recognized. Should be one of AE, noAuto_noSigma, noAuto, auto, or flow.'%model_type)

    print('Parameters for VAE model type %s: '%model_type, params_vae)

    #Define potential energy function
    #This is also specific to the system at hand
    if 'BAT' in data_file:
        prmtopFile = os.path.expanduser('~/Autoencoders_Latent/Ala_Dipeptide/sim_files/alanine-dipeptide.prmtop')
        strucFile = os.path.expanduser('~/Autoencoders_Latent/Ala_Dipeptide/sim_files/alanine-dipeptide.pdb')
        uni = mda.Universe(prmtopFile, strucFile)
        bat_analysis = BAT_tf(uni.select_atoms('all'))
        tfunc = transform_ala_tf(bat_to_cart_func=bat_analysis.Cartesian_tf)
        efunc = losses.dialanineHamiltonian()
        def pot_func(coords):
            new_coords = tfunc(coords)
            return efunc(new_coords)

    else:
        def tfunc(xyz):
            return np.reshape(xyz, (-1, 3))

        pot_func = losses.dialanineHamiltonian(transform_func=tfunc)

    #Loop over latent sizes
    for l_dim in latent_sizes:

        #Loop over beta values
        for bval in beta_vals:

            #Load the model
            vae_model = vae_class(raw_data.shape[1:],
                                  l_dim,
                                  beta=bval,
                                  **params_vae)
            unused_recon = vae_model(raw_data[:2])
            del unused_recon
            unused_flow = vae_model.flow(tf.random.normal((2, l_dim)))
            del unused_flow
            vae_model.load_weights('%s/%s_l%i_B%i.h5'%(base_dir, model_type, l_dim, bval))

            #Define draw type based on beta value
            #Can't always trust flow for beta of 0, so use draw direct
            if bval == 0:
                draw_type = 'direct'
            else:
                draw_type = 'std_normal'

            #Run MC simulation
            out_conf, out_U, traj, traj_pot, num_acc, mc_stats = run_mc_sim(raw_data,
                                                                            pot_func,
                                                                            vae_model,
                                                                          draw_type=draw_type,
               num_steps=1000, write_freq=10,
               num_parallel=1000,
                                                                           )
            #Save all MC info to single npz file
            np.savez('%s/mc_%s_l%i_B%i.npz'%(base_dir, model_type, l_dim, bval),
                     traj=traj,
                     U=traj_pot,
                     mc_stats=mc_stats)

            gc.collect()


if __name__ == '__main__':
    try:
        dir_name = sys.argv[3]
    except IndexError:
        dir_name = '.'
    main(sys.argv[1], sys.argv[2], base_dir=dir_name)

