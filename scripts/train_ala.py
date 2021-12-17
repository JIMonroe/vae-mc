from libVAE import dataloaders, vae, train


#First train "naturally" by converting to sine-cosine pairs as part of the encoding
#and then decoding to a periodic von Mises distribution
#This requires defining periodic degress of freedom, which here are just the torsions
#In the MDAnalysis BAT coordinates, these are the last N-3 DOFs (N is the number of atoms)
#For alanine, have N=22, so have 19 torsions
periodic_inds = list(range(-19, 0))

#And also remember that we are ignoring translation and rotation, so lose 6 DOFs
#And we also constrain bonds involving hydrogens, which is 12 of them
#(all handled during data processing in dataloaders, but means that output_dim is 66-12-6 = 48

#Load in data with periodic degrees of freedom (not sine-cosine pairs)
data_file = 'ala_dipeptide_BAT.npy'
batch_size = 200
trainData, valData = dataloaders.ala_dipeptide_data(data_file, batch_size, val_frac=0.05, rigid_bonds=True, sin_cos=False)

#Train with only 1 latent dimension
vaeModel = vae.PriorFlowVAE((48,), 1, include_vars=True, autoregress=True, n_auto_group=1,
                            periodic_dof_inds=periodic_inds,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l1', overwrite=True, anneal_beta_val=1.0)
#To extend training, comment out fist training call (with annealing) and uncomment line below
#vaeModel.load_weights('trained_models/pflow_auto_periodic_BAT_l1/training.ckpt')
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l1', overwrite=True)

# #5 latent dimensions (should theoretically capture backbone dihedrals fully b/c >4)
# del vaeModel
# vaeModel = vae.PriorFlowVAE((48,), 5, include_vars=True, autoregress=True, n_auto_group=1,
#                             periodic_dof_inds=periodic_inds,
#                             e_hidden_dim=300,
#                             d_hidden_dim=300)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l5', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l5', overwrite=True)

#10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((48,), 10, include_vars=True, autoregress=True, n_auto_group=1,
                            periodic_dof_inds=periodic_inds,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_periodic_BAT_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_periodic_BAT_l10', overwrite=True)

#20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((48,), 20, include_vars=True, autoregress=True, n_auto_group=1,
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
trainData, valData = dataloaders.ala_dipeptide_data(data_file, batch_size, val_frac=0.05, rigid_bonds=True, sin_cos=True)

#Dimension of output will now be increased by 19 since torsion DOFs doubled
#Train with only 1 latent dimension
vaeModel = vae.PriorFlowVAE((67,), 1, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l1', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l1', overwrite=True)

# #5 latent dimensions
# del vaeModel
# vaeModel = vae.PriorFlowVAE((67,), 5, include_vars=True, autoregress=True, n_auto_group=1,
#                             e_hidden_dim=300,
#                             d_hidden_dim=300)
# vaeModel.beta = 0.0
# train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l5', overwrite=True, anneal_beta_val=1.0)
# train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l5', overwrite=True)

#10 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((67,), 10, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l10', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l10', overwrite=True)

#20 latent dimensions
del vaeModel
vaeModel = vae.PriorFlowVAE((67,), 20, include_vars=True, autoregress=True, n_auto_group=1,
                            e_hidden_dim=300,
                            d_hidden_dim=300)
vaeModel.beta = 0.0
train.trainCustom(vaeModel, trainData, valData, num_epochs=40, save_dir='pflow_auto_sincos_BAT_l20', overwrite=True, anneal_beta_val=1.0)
train.trainCustom(vaeModel, trainData, valData, num_epochs=160, save_dir='pflow_auto_sincos_BAT_l20', overwrite=True)


