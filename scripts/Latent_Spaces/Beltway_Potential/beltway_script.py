
import sys
import gc
from datetime import datetime
import numpy as np
import tensorflow as tf

from libVAE import vae, train

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
            #But only if at beta value less than or equal to 1
            #If at beta of zero, will delete model at end
            #If at 1, get fresh model and train
            #Then for 2, etc. re-use old model, but new saved files, and just anneal
            #to the next beta value from the old one
            if bval <= 1:
                vae_model = vae_class(raw_data.shape[1:],
                                      l_dim,
                                      beta=0.0,
                                      **params_vae)

            this_file_name = save_file_prefix+'_l%i_B%i'%(l_dim, bval)
            print('\n\nTraining model %s'%this_file_name)
            #Train this model
            loss_info = train.trainRawData(vae_model,
                                           raw_data,
                                           this_file_name+'.h5',
                                           num_epochs=100,
                                           batch_size=200,
                                           anneal_beta_val=bval,
                                           anneal_epochs=20)
            #Save loss information
            np.savetxt('loss_info_'+this_file_name+'.txt', loss_info,
                       header='Loss    Recon    KL    Val_Loss    Val_Recon    Val_KL')

            #If beta is zero or have regular AE, must train flow separately after rest
            #(not that beta should be zero for AE for models trained here)
            if bval == 0.0 or model_type == 'AE':
                train.trainFlowOnly(vae_model,
                                    raw_data,
                                    this_file_name+'.h5',
                                    num_epochs=100,
                                    batch_size=200) 

            #Clean up
            if bval == 0.0:
                del vae_model
            gc.collect()

            print('Model training complete at %s'%str(datetime.now()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


