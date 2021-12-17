from libVAE import dataloaders, vae, train


#First train "naturally" by converting to sine-cosine pairs as part of the encoding
#and then decoding to a periodic von Mises distribution
#This requires defining periodic degress of freedom, which here are just the torsions
#In the MDAnalysis BAT coordinates, these are the last N-3 DOFs (N is the number of atoms)
#For our linear polymer with N=20, have 17 torsions
periodic_inds = list(range(-17, 0))

#And also remember that we are ignoring translation and rotation, so lose 6 DOFs
#And we also constrain bonds, so ignore those 19 DOFs
#(all handled during data processing in dataloaders, but means that output_dim is 60-19-6 = 35

#Load in data without sine-cosine pair conversion
data_file = 'polymer_BAT.npy'
batch_size = 200
trainData, valData = dataloaders.polymer_data(data_file, batch_size, val_frac=0.05, rigid_bonds=True, sin_cos=False)

#Train with only 1 latent dimension
vaeModel = vae.PriorFlowVAE((35,), 1, include_vars=True, autoregress=True, n_auto_group=1,
                            periodic_dof_inds=periodic_inds,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l1', overwrite=True)

#5 latent dimensions
# del vaeModel
# vaeModel = vae.PriorFlowVAE((35,), 5, include_vars=True, autoregress=True, n_auto_group=1,
#                             periodic_dof_inds=periodic_inds,
#                             e_hidden_dim=300,
#                             d_hidden_dim=300)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l5', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l5', overwrite=True)

#10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((35,), 10, include_vars=True, autoregress=True, n_auto_group=1,
                            periodic_dof_inds=periodic_inds,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l10', overwrite=True)

#20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((35,), 20, include_vars=True, autoregress=True, n_auto_group=1,
                            periodic_dof_inds=periodic_inds,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l20', overwrite=True)


#Now train with data processing converting to sine-cosine pairs and decoder predicting pairs
#as well
del vaeModel
del trainData, valData
trainData, valData = dataloaders.polymer_data(data_file, batch_size, val_frac=0.05, rigid_bonds=True, sin_cos=False)

#Dimension of output will now be increased by 17 since torsion DOFs doubled
#Train with only 1 latent dimension
vaeModel = vae.PriorFlowVAE((52,), 1, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l1', overwrite=True)

# #5 latent dimensions
# del vaeModel
# vaeModel = vae.PriorFlowVAE((52,), 5, include_vars=True, autoregress=True, n_auto_group=1,
#                             e_hidden_dim=300,
#                             d_hidden_dim=300)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l5', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l5', overwrite=True)

#10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((52,), 10, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l10', overwrite=True)

#20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((52,), 20, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l20', overwrite=True)


