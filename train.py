# Written in 2020 by Jacob I. Monroe, NIST Employee

"""Functions for training a VAE model, including compiling models.
"""

import os
import time
from libVAE import dataloaders, losses, vae
import numpy as np
import tensorflow as tf


def compileModel(model):
  """Compiles a model with a defined optimizer and loss function.
  """

  #Create an optimizer to use based on hyperparameters in https://github.com/google-research/disentanglement_lib 
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  #Now compile the model so it's ready to train
  model.compile(optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.SUM),
                #loss=losses.ReconLoss(),
                #loss=losses.TotalVAELoss(model),
                #metrics=[losses.ReconLossMetric(),
                #         losses.ElboLossMetric(model),
                #         losses.TotalLossMetric(model)],
               )

  #Should be acting on model object passed in, but to be safe, return the compiled model
  return model


def train(model,
          data_file,
          num_epochs=2,
          batch_size=64,
          save_dir='vae_info',
          overwrite=False):
  """Trains a VAE model and saves the results in a way that the model can be fully reloaded.

  Args:
    model: the VAE model object to train and save
    data_file: a file containing the data for training/validation
    num_epochs: Integer with number of epochs to train (each is over all samples)
    batch_size: Integer with the batch size
    save_dir: String with path to directory to save to
    overwrite: Boolean determining whether data is overwritten
  """

  #Compile the model - this will reset the optimizer and loss functions
  #Otherwise, does not affect the model in any way
  #Probably good to restart optimizer each time train on new data
  model = compileModel(model)

  #Check if the directory exists
  #If so, assume continuation and load model weights
  if os.path.isdir(save_dir):
    #If overwrite is True, don't need to do anything
    #If it's False, create a new directory to save to
    if not overwrite:
      print("Found saved model at %s and overwrite is False."%save_dir)
      print("Will attempt to load and continue training.")
      model.load_weights(os.path.join(save_dir, 'training.ckpt'))
      #print(model.summary())
      try:
        dir_split = save_dir.split('_')
        train_num = int(dir_split[-1])
        save_dir = '%s_%i'%("_".join(dir_split[:-1]), train_num+1)
      except ValueError:
        save_dir = '%s_1'%(save_dir)
      os.mkdir(save_dir)
  #If not, create that directory to save to later
  else:
    os.mkdir(save_dir)

  print("Model set up and ready to train.")

  #Want to load in data
  #Can actually just give numpy arrays to the Model.fit function in tf.keras
  #So could just load .nc files directly
  #Would still like to provide a wrapper in dataloaders.py
  #Will make more generalizable in case data format changes
  #But, something weird with batching happens if you use keras loss functions
  #trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.1)
  #trainData = dataloaders.raw_image_data(data_file)
  trainData, valData = dataloaders.dsprites_data(batch_size, val_frac=0.01)

  #Set up path for checkpoint files
  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #We will want to save checkpoints, so set this up
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   verbose=2,
                                                   save_freq='epoch',
                                                   save_weights_only=True,
                                                  )
  model.save_weights(checkpoint_path.format(epoch=0))

  print("Beginning training at: %s"%time.ctime())

  #Train the model
  train_history = model.fit(trainData,
                            #trainData,
                            validation_data=valData,
                            #batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=2,
                            callbacks=[cp_callback],
                            #validation_split=0.10,
                            shuffle=True,
                           )

  print("Training completed at: %s"%time.ctime())
  print(model.summary())

  #print(train_history.history)

  #At the end, save the full model
  #This is of limited usefulness due to difficulties saving and loading custom models
  #If can manage not to have any custom losses or metrics, loads just fine!
  #Well, fine except that it may lose the regularizer function
  model.save(os.path.join(save_dir, 'model'), save_format='tf')


