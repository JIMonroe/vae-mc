
import sys
import gc
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf

from libVAE import vae, analysis

#Try to set GPU to allow multiple processes/trainings
#Assumes just have one GPU, so still set CUDA_VISIBLE_DEVICES
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def main(data_file, save_file_prefix):
    #Define key VAE parameters and name
    params_vae = {'e_hidden_dim':50,
                  'f_hidden_dim':20,
                  'd_hidden_dim':50,}
    vae_class = vae.PriorFlowVAE
    beta_vals = [0.0, 1.0, 2.0, 5.0, 10.0]
    latent_sizes = [1, 2]

    #Load in data - MUST be loadable through np.load, so any preprocessing should be applied
    raw_data = np.load(data_file)

    #Specify periodic DOFs for specific system - CHANGE BASED ON SYSTEM!
    if 'rtheta' in data_file:
        params_vae['periodic_dof_inds'] = [1,]

    #Define model and parameters based on the file prefix
    model_type = save_file_prefix.split('/')[-1]
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

    #Loop over latent space sizes
    for l_dim in latent_sizes:

        #And loop over the beta values
        for bval in beta_vals:

            #Create our model
            vae_model = vae_class(raw_data.shape[1:],
                                  l_dim,
                                  beta=bval,
                                  **params_vae)

            #If bval is set to zero, won't load flow params, but want them
            #Value itself only matters for training and loading model params
            if bval == 0.0:
                vae_model.beta = 1.0

            this_file_name = save_file_prefix+'_l%i_B%i'%(l_dim, bval)
            print('\n\nAnalysis for model %s'%this_file_name)

            #Pass sample through to initialize so can load parameters correctly
            unused_recon = vae_model(raw_data[:2])
            del unused_recon
            unused_flow = vae_model.flow(tf.random.normal((2, l_dim)))
            del unused_flow

            #Now can load parameters
            vae_model.load_weights('%s.h5'%this_file_name)

            #Perform analysis after initiating object
            #Do without bootstrap and print info, then do with bootstrap
            latent_analysis = analysis.LatentAnalysis(vae_model, raw_data)
            latent_analysis.analyze(n_per=100)
            print('Result without bootstrap: ')
            print(latent_analysis.results)
            # latent_analysis.analyze(bootstrap=True, n_boot=100)

            #Save dictionary as pickle file
            #Don't want to save whole LatentAnalysis object because references vae model
            #Need to do because flows are necessary for some analyses
            #But don't want to try to pickle that
            with open('analysis_%s.pkl'%this_file_name, 'wb') as pfile:
                pickle.dump(latent_analysis.results, pfile)

            #Clean up to open up memory
            del latent_analysis
            del vae_model
            gc.collect()

            print('Model analysis complete at %s'%str(datetime.now()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


