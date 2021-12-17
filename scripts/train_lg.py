import glob
import tensorflow as tf
from libVAE import dataloaders, vae, train

dataFiles = glob.glob('f*random/*.nc')

#Load in training data
batch_size = 200
trainData, valData = dataloaders.image_data(dataFiles, batch_size, val_frac=0.05)

#Specify loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                           reduction=tf.keras.losses.Reduction.SUM)

#First train the best model
vaeModel = vae.PriorFlowVAE((28, 28, 1), 1, include_vars=False, autoregress=True, use_skips=True, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_auto_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_auto_l1', overwrite=True)

#Attempt to clear up some memory
del vaeModel

#Next train the CG model in two stages
#First train the autoencoder, then train the flow
#Can accomplish by setting beta to 0, then switch to 1
cgModel = vae.CGModel((28, 28, 1), 'lg', autoregress=True, use_skips=True)
cgModel.beta = 0.0
train.trainCustom(cgModel, trainData, valData, loss_fn, num_epochs=80, save_dir='cgmodel_auto', overwrite=True)
# cgModel.load_weights('trained_models/cgmodel_auto/training.ckpt')
cgModel.beta = 1.0
train.trainCustom(cgModel, trainData, valData, loss_fn, num_epochs=20, save_dir='cgmodel_auto', overwrite=True)
del cgModel

#Train a special CGModel mapping to a lattice gas with a reduced grid size
#Since flow is much more complicated here, train it the whole time with the decoder
cgModel = vae.CGModel_AutoPrior((28, 28, 1), autoregress=True, use_skips=True, num_latent=49)
train.trainCustom(cgModel, trainData, valData, loss_fn, num_epochs=100, save_dir='reduceMap_cgmodel_auto', overwrite=True)
del cgModel

#Train the Srel model, also in two stages
#Technically replicating same thing for first stage of CGModel but good to check
srelModel = vae.SrelModel((28, 28, 1), 'lg', autoregress=True, use_skips=True)
train.trainSrelCG(srelModel, dataFiles, num_epochs=10, batch_size=200, save_dir='srel_auto', overwrite=True, mc_beta=1.77)
train.trainCustom(srelModel, trainData, valData, loss_fn, num_epochs=80, save_dir='srel_auto', overwrite=True)
# srelModel.load_weights('trained_models/srel_auto/training.ckpt')
train.trainSrelCG(srelModel, dataFiles, num_epochs=10, batch_size=100, save_dir='srel_auto', overwrite=True, mc_beta=1.77)
del srelModel

#Train different Srel model, specifically a reduction to a smaller grid
#Here with a stochastic mapping (average of sites is used to draw new site)
srelModel = vae.SrelModel((28, 28, 1), 'lg', autoregress=True, use_skips=True,
                          num_latent=49,
                          cg_map_info={'n_group':4, 'sample':True, 'sample_stochastic':True})
#train.trainSrelCG(srelModel, trainData, valData, loss_fn, num_epochs=10, save_dir='reduceMap_srel_auto', overwrite=True, mc_beta=1.77)
srelModel.load_weights('reduceMap_srel_auto/training.ckpt')
train.trainCustom(srelModel, trainData, valData, loss_fn, num_epochs=80, save_dir='reduceMap_srel_auto', overwrite=True)
train.trainSrelCG(srelModel, dataFiles, num_epochs=10, batch_size=200, save_dir='reduceMap_srel_auto', overwrite=True, mc_beta=1.77)
del srelModel

#Next train the VAE model with no skips
vaeModel = vae.PriorFlowVAE((28, 28, 1), 1, include_vars=False, autoregress=True, use_skips=False, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_auto_noskips_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_auto_noskips_l1', overwrite=True)

#Now train the model with no autoregression
del vaeModel
vaeModel = vae.PriorFlowVAE((28, 28, 1), 1, include_vars=False, autoregress=False, use_skips=False, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_noauto_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_noauto_l1', overwrite=True)

#Train a model with the full package but now with a latent dimension of 10
del vaeModel
vaeModel = vae.PriorFlowVAE((28, 28, 1), 10, include_vars=False, autoregress=True, use_skips=True, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_auto_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_auto_l10', overwrite=True)

#Latent dimension of 10 with no skips
del vaeModel
vaeModel = vae.PriorFlowVAE((28, 28, 1), 10, include_vars=False, autoregress=True, use_skips=False, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_auto_noskips_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_auto_noskips_l10', overwrite=True)

#Latent dimension 10 with no autoregression
del vaeModel
vaeModel = vae.PriorFlowVAE((28, 28, 1), 10, include_vars=False, autoregress=False, use_skips=False, n_auto_group=1)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='pflow_noauto_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='pflow_noauto_l10', overwrite=True)


#################################################################################
#Below is retraining based on VAE-based MC sims
#Should have done above training, then run VAE-based MC sims based on models, then run below
#Mixed means also adding in other types of MC moves
#The idea was to see if can keep good acceptance rate when retraining on VAE-based MC data
#Found, however, that what's really needed is just more steps between saves
#This allows for more independent configurations in the same way that mixing does
#See below
#################################################################################
# retrain_dataFiles = ['mc_acceptance/allData_l1/LG_2D_traj.nc',
#                      'mc_acceptance/allData_l1_mixedMC/LG_2D_traj.nc']
# trainData, valData = dataloaders.image_data(retrain_dataFiles, batch_size, val_frac=0.05)
# #Look at retraining best 1D and 10D latent models on data generated by MC with previous VAE
# vaeModel = vae.PriorFlowVAE((28, 28, 1), 1, include_vars=False, autoregress=True, use_skips=True, n_auto_group=1)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='mixData_retrained_pflow_auto_l1', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='mixData_retrained_pflow_auto_l1', overwrite=True) 
# 
# del vaeModel
# trainData, valData = dataloaders.image_data('mc_acceptance/allData_l10/LG_2D_traj.nc', batch_size, val_frac=0.05)
# vaeModel = vae.PriorFlowVAE((28, 28, 1), 10, include_vars=False, autoregress=True, use_skips=True, n_auto_group=1)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='retrained_pflow_auto_l10', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=80, save_dir='retrained_pflow_auto_l10', overwrite=True)
# 
# #Here also retrained, but retrained on longer-spaced saves from original VAE-based MC runs
# #And using separate validation data to better monitor overfitting during training
# dataFiles = glob.glob('mc_acceptance/save100x_allData_l1/segment*/LG_2D_traj.nc')
# valFiles = glob.glob('f*random/*.nc')
# 
# trainData, unusedData = dataloaders.image_data(data_file, batch_size, val_frac=0.01)
# unusedData, valData = dataloaders.image_data(val_file, batch_size, val_frac=0.2)
# del unusedData
# 
# #Note that as we train, also saving model at various points so can check for overfitting
# #That is, check entire model rather than just reported validation loss metrics
# vaeModel = vae.PriorFlowVAE((28, 28, 1), 1, include_vars=False, autoregress=True, use_skips=True, n_auto_group=1)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=20, save_dir='save100x_sepVal0_retrained_pflow_auto_l1', overwrite=True, anneal_beta_val=1.0, val_file=valFiles)
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=30, save_dir='save100x_sepVal30_retrained_pflow_auto_l1', overwrite=True, val_file=valFiles)
# train.trainCustom(vaeModel, trainData, valData, loss_fn, num_epochs=50, save_dir='save100x_sepVal80_retrained_pflow_auto_l1', overwrite=True, val_file=valFiles)

