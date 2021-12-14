
import numpy as np
import tensorflow as tf
from libVAE import dataloaders, vae, train


#Load in data - here without preoprocesing to center and whiten
data_file = 'trajdata_dimer.npz'
batch_size = 200
trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
                                           dset='all', permute=True, center_and_whiten=False)

#Train with 1 latent dimension
#n_auto_group is 2 because makes sense with 2D points, though doesn't affect training much
vaeModel = vae.PriorFlowVAE((76,), 1, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_l1', overwrite=True)

#And with 10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((76,), 10, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_l10', overwrite=True)

#And 20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((76,), 20, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_l20', overwrite=True)


#Do again with centering and whitening, but must run separately after modifying train.py
del vaeModel
del trainData, valData
trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
                                           dset='all', permute=True, center_and_whiten=True)

#Train with 1 latent dimension
#n_auto_group is 2 because makes sense with 2D points, though doesn't affect training much
vaeModel = vae.PriorFlowVAE((76,), 1, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_centerwhite_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_centerwhite_l1', overwrite=True)

#And with 10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((76,), 10, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_centerwhite_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_centerwhite_l10', overwrite=True)

#And 20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((76,), 20, include_vars=True, autoregress=True, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_centerwhite_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_centerwhite_l20', overwrite=True)


#Alternatively with full flow on prior AND decoder
vaeModel = vae.FullFlowVAE((76,), 1, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='fullflow_auto_centerwhite_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='fullflow_auto_centerwhite_l1', overwrite=True)

#And with 10 latent dimensions
del vaeModel
vaeModel = vae.FullFlowVAE((76,), 10, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='fullflow_auto_centerwhite_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='fullflow_auto_centerwhite_l10', overwrite=True)

#And 20 latent dimensions
del vaeModel
vaeModel = vae.FullFlowVAE((76,), 20, n_auto_group=2,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='fullflow_auto_centerwhite_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='fullflow_auto_centerwhite_l20', overwrite=True)


