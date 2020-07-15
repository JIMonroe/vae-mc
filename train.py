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
  trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.1)
  #trainData = dataloaders.raw_image_data(data_file)
  #trainData, valData = dataloaders.dsprites_data(batch_size, val_frac=0.01)

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
                            #validation_data=valData,
                            #batch_size=batch_size,
                            epochs=num_epochs,
                            verbose=2,
                            #callbacks=[cp_callback],
                            #validation_split=0.10,
                            #shuffle=True,
                           )

  print("Training completed at: %s"%time.ctime())
  print(model.summary())

  model.save_weights(checkpoint_path.format(epoch=num_epochs))

  #print(train_history.history)

  #At the end, save the full model
  #This is of limited usefulness due to difficulties saving and loading custom models
  #If can manage not to have any custom losses or metrics, loads just fine!
  #Well, fine except that it may lose the regularizer function
  model.save(os.path.join(save_dir, 'model'), save_format='tf')


def trainCustom(model,
                data_file,
                num_epochs=2,
                batch_size=64,
                save_dir='vae_info',
                overwrite=False,
                extraLossFunc=None,
                extraLossWeight=1.0):
  """Trains a VAE model and saves the results in a way that the model can be fully reloaded.
Uses a custom training loop rather than those built into the tf.keras.Model class.

  Args:
    model: the VAE model object to train and save
    data_file: a file containing the data for training/validation
    num_epochs: Integer with number of epochs to train (each is over all samples)
    batch_size: Integer with the batch size
    save_dir: String with path to directory to save to
    overwrite: Boolean determining whether data is overwritten
    extraLossFunc: Additional loss function to add (for example potential energies)
  """

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
  #trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.05)
  trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
                                                 dset='all', permute=True)#, center_and_whiten=True)
  #trainData = dataloaders.raw_image_data(data_file)
  #trainData, valData = dataloaders.dsprites_data(batch_size, val_frac=0.01)

  #Set up path for checkpoint files
  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Create an optimizer to use based on hyperparameters in https://github.com/google-research/disentanglement_lib 
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  #Specify the loss function we want to use
  #loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  #                                reduction=tf.keras.losses.Reduction.SUM)
  #loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
  #loss_fn = losses.ReconLoss()
  #loss_fn = losses.diag_gaussian_loss
  loss_fn = losses.ReconLoss(loss_fn=losses.diag_gaussian_loss, activation=None,
                             reduction=tf.keras.losses.Reduction.SUM)

  print("Beginning training at: %s"%time.ctime())

  #Loop over epochs
  for epoch in range(num_epochs):
    print('\nOn epoch %d:'%epoch)

    #Iterate over batches in the dataset
    for step, x_batch_train in enumerate(trainData):
      for ametric in model.metrics:
        ametric.reset_states()
      with tf.GradientTape() as tape:
        reconstructed = model(x_batch_train[0])
        loss = loss_fn(x_batch_train[0], reconstructed) / x_batch_train[0].shape[0]
        loss += sum(model.losses)
        if extraLossFunc is not None:
          extra_loss = tf.cast(extraLossFunc(x_batch_train[0], reconstructed), 'float32')
          loss += extraLossWeight*extra_loss
        else:
          extra_loss = 0.0

      grads = tape.gradient(loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step%100 == 0:
        print('\tStep %i: loss=%f, model_loss=%f, kl_div=%f, reg_loss=%f, extra_loss=%f'
              %(step, loss, sum(model.losses), 
                model.metrics[0].result(), model.metrics[1].result(), extra_loss))

    #Save checkpoint after each epoch
    print('\tEpoch finished, saving checkpoint.')
    model.save_weights(checkpoint_path.format(epoch=epoch))

    #Check against validation data
    val_loss = tf.constant(0.0)
    val_extra_loss = tf.constant(0.0)
    for ametric in model.metrics:
      ametric.reset_states()
    batchCount = 0.0
    for x_batch_val in valData:
      reconstructed = model(x_batch_val[0])
      val_loss += loss_fn(x_batch_val[0], reconstructed) / x_batch_val[0].shape[0]
      val_loss += sum(model.losses)
      if extraLossFunc is not None:
        extra_loss = tf.cast(extraLossFunc(x_batch_val[0], reconstructed), 'float32')
        val_loss += extraLossWeight*extra_loss
        val_extra_loss += extra_loss
      batchCount += 1.0
    val_loss /= batchCount
    val_extra_loss /= batchCount
    print('\tValidation loss=%f, model_loss=%f, kl_div=%f, reg_loss=%f, extra_loss=%f'
          %(val_loss, sum(model.losses), 
            model.metrics[0].result(), model.metrics[1].result(), val_extra_loss))

  print("Training completed at: %s"%time.ctime())
  print(model.summary())

  #print(train_history.history)

  #At the end, save the full model
  #This is of limited usefulness due to difficulties saving and loading custom models
  #If can manage not to have any custom losses or metrics, loads just fine!
  #Well, fine except that it may lose the regularizer function
  #model.save(os.path.join(save_dir, 'model'), save_format='tf')


#This approach DOES NOT WORK
#The method of performing averaging implemented below will not correctly compute the derivative
#Because the probability of an x configuration itself depends on the parameters,
#simple averaging over generated z and x will not work.
#Even with reparametrization tricks the issue will remain.
def trainGenerator(model,
                   num_epochs=10000,
                   batch_size=64,
                   z_batch_size=1000,
                   save_dir='vae_info',
                   overwrite=False,
                   beta=2.0,
                   mu=-2.0,
                   eps=-1.0):
  """Trains only the decoder/generator portion of a VAE model to reproduce relative
Boltzmann weights (specific to the loss function used) starting from standard normal
distributions of latent variables.

  Args:
    model: the VAE model object to train and save
    num_epochs: Integer with number of epochs to train (here training steps)
    batch_size: Integer with number of samples to generate and use per training step
    save_dir: String with path to directory to save to
    overwrite: Boolean determining whether data is overwritten
    beta: inverse temperature of ensemble to train towards
    mu: chemical potential of lattice gas
    eps: interaction energy of lattice gas
  """

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

  print("Model set up and ready to train in terms of generation.")

  #Set up path for checkpoint files
  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Create an optimizer to use based on hyperparameters in https://github.com/google-research/disentanglement_lib 
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  print("Beginning training at: %s"%time.ctime())

  #Loop over epochs
  for epoch in range(num_epochs):

    #First step is to generate z sample from the VAE model
    #If the model is well-trained, the model distribution of z should be standard normal
    z_sample = tf.random.normal((z_batch_size, model.num_latent))

    #Obtain probabilities of x given z
    #x_logits_temp = model.decoder(z_sample)
    #x_probs_temp = tf.math.sigmoid(x_logits_temp)

    #Generate x samples from probabilities given z samples
    #rand_inds = np.random.choice(x_probs_temp.shape[0], size=batch_size, replace=False)
    #rand_probs = tf.random.uniform(x_probs_temp.shape)
    #x_sample = tf.cast((x_probs_temp > rand_probs), 'float32').numpy()[rand_inds]

    #Randomly select some x probabilities to use as samples
    rand_inds = np.random.choice(z_batch_size, size=batch_size, replace=False)
 
    #Compute loss based on generated sample
    with tf.GradientTape() as tape:
      #Obtain probabilities of x given z
      x_logits = model.decoder(z_sample)
      x_probs = tf.math.sigmoid(x_logits)

      #To get x samples, should technically draw based on x_probs
      #However, then we cannot take derivatives with respect to the model parameters
      #This also means the computed potential energies will be the average energies
      #Technically still taking gradient of an expectation that depends on model parameters
      #Sort of funky
      x_sample = tf.gather(x_probs, rand_inds, axis=0)

      loss, loss_px, loss_u = losses.relative_boltzmann_loss(x_sample, x_probs,
                                                             beta=beta, func_params=[mu, eps])

    grads = tape.gradient(loss, model.decoder.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.decoder.trainable_weights))

    if epoch%100 == 0:
      print('\nStep %i: loss=%f, log_px=%f, u=%f'%(epoch, loss, loss_px, loss_u))
      #Save checkpoint
      print('\tSaving checkpoint.')
      model.save_weights(checkpoint_path.format(epoch=epoch))

  print("Training completed at: %s"%time.ctime())
  model.save_weights(checkpoint_path.format(epoch=num_epochs-1))
  #print(model.summary())


def trainAdversarial(model,
                     data_file,
                     num_epochs=2,
                     num_dis_loops=5,
                     batch_size=64,
                     save_dir='vae_info',
                     overwrite=False,
                     extraLossFunc=None,
                     extraLossWeight=1.0):
  """Trains adversarial VAE model, which requires a completely different training loop
than for a standard VAE. Only a couple of extra parameters are needed to describe how
the discriminator network should be trained.

  Args:
    model: the VAE model object to train and save
    data_file: a file containing the data for training/validation
    num_epochs: Integer with number of epochs to train (each is over all samples)
    num_dis_loops: Number of training loops for discriminator for each batch of samples
    batch_size: Integer with the batch size
    save_dir: String with path to directory to save to
    overwrite: Boolean determining whether data is overwritten
    extraLossFunc: Additional loss function to add (for example potential energies)
  """

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
  #trainData, valData = dataloaders.image_data(data_file, batch_size, val_frac=0.05)
  trainData, valData = dataloaders.dimer_2D_data(data_file, batch_size, val_frac=0.05,
                                                 dset='all', permute=True)#, center_and_whiten=True)
  #trainData = dataloaders.raw_image_data(data_file)
  #trainData, valData = dataloaders.dsprites_data(batch_size, val_frac=0.01)

  #Before regular training loop, need to get discriminator really well trained
  #Important for validity of the method
  #Will use separate optimizer for this
  dis_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           epsilon=1e-08,
                                          )

  print("\nInitial discriminator training...")
  for epoch in range(2):
    print("On epoch %d:"%epoch)

    #Iterate over batches in dataset
    for step, x_batch_train in enumerate(trainData):
      z_batch = model.encoder(x_batch_train[0])
      z_prior = tf.random.normal(z_batch.shape)
      with tf.GradientTape() as tape:
        logits_batch = model.discriminator(x_batch_train[0], z_batch)
        logits_prior = model.discriminator(x_batch_train[0], z_prior)
        dis_loss = tf.keras.losses.binary_crossentropy(tf.ones(logits_batch.shape),
                                                       logits_batch,
                                                       from_logits=True)
        dis_loss += tf.keras.losses.binary_crossentropy(tf.zeros(logits_prior.shape),
                                                        logits_prior,
                                                        from_logits=True)
      dis_grads = tape.gradient(dis_loss, model.discriminator.trainable_weights)
      dis_optimizer.apply_gradients(zip(dis_grads, model.discriminator.trainable_weights))

      if step%100 == 0:
        print('\tStep %i: dis_loss: %f'%(step, dis_loss))
  print("Finished with initial discriminator training.")

  #Set up path for checkpoint files
  checkpoint_path = os.path.join(save_dir, 'training.ckpt')

  #Create an optimizer to use based on hyperparameters in https://github.com/google-research/disentanglement_lib 
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=1e-08,
                                      )

  #Specify the loss function we want to use
  #loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  #                                reduction=tf.keras.losses.Reduction.SUM)
  #loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
  #loss_fn = losses.ReconLoss()
  #loss_fn = losses.diag_gaussian_loss
  loss_fn = losses.ReconLoss(loss_fn=losses.diag_gaussian_loss, activation=None,
                             reduction=tf.keras.losses.Reduction.SUM)

  print("Beginning training at: %s"%time.ctime())

  #Loop over epochs
  for epoch in range(num_epochs):
    print('\nOn epoch %d:'%epoch)

    #Iterate over batches in the dataset
    for step, x_batch_train in enumerate(trainData):
      with tf.GradientTape(persistent=True) as tape:
        z_batch = model.encoder(x_batch_train[0])
        z_prior = tf.random.normal(z_batch.shape)
        for j in range(num_dis_loops):
          logits_batch = model.discriminator(x_batch_train[0], z_batch)
          logits_prior = model.discriminator(x_batch_train[0], z_prior)
          dis_loss = tf.keras.losses.binary_crossentropy(tf.ones(logits_batch.shape),
                                                         logits_batch,
                                                         from_logits=True)
          dis_loss += tf.keras.losses.binary_crossentropy(tf.zeros(logits_prior.shape),
                                                          logits_prior,
                                                          from_logits=True)
          with tape.stop_recording():
            dis_grads = tape.gradient(dis_loss, model.discriminator.trainable_weights)
            dis_optimizer.apply_gradients(zip(dis_grads, model.discriminator.trainable_weights))
        reconstructed = model.decoder(z_batch)
        loss = loss_fn(x_batch_train[0], reconstructed) / x_batch_train[0].shape[0]
        kl_loss = tf.reduce_mean(logits_batch)
        loss += model.beta*kl_loss
        if extraLossFunc is not None:
          extra_loss = tf.cast(extraLossFunc(x_batch_train[0], reconstructed), 'float32')
          loss += extraLossWeight*extra_loss
        else:
          extra_loss = 0.0

      grads = tape.gradient(loss, model.encoder.trainable_weights+model.decoder.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.encoder.trainable_weights+model.decoder.trainable_weights))
      del tape

      if step%100 == 0:
        print('\tStep %i: loss=%f, kl_div=%f, reg_loss=%f, dis_loss=%f, extra_loss=%f'
              %(step, loss, kl_loss, model.beta*kl_loss, dis_loss, extra_loss))

    #Save checkpoint after each epoch
    print('\tEpoch finished, saving checkpoint.')
    model.save_weights(checkpoint_path.format(epoch=epoch))

    #Check against validation data
    val_loss = tf.constant(0.0)
    val_extra_loss = tf.constant(0.0)
    for ametric in model.metrics:
      ametric.reset_states()
    batchCount = 0.0
    for x_batch_val in valData:
      reconstructed = model(x_batch_val[0])
      val_loss += loss_fn(x_batch_val[0], reconstructed) / x_batch_val[0].shape[0]
      val_loss += sum(model.losses)
      if extraLossFunc is not None:
        extra_loss = tf.cast(extraLossFunc(x_batch_val[0], reconstructed), 'float32')
        val_loss += extraLossWeight*extra_loss
        val_extra_loss += extra_loss
      batchCount += 1.0
    val_loss /= batchCount
    val_extra_loss /= batchCount
    print('\tValidation loss=%f, model_loss=%f, kl_div=%f, reg_loss=%f, extra_loss=%f'
          %(val_loss, sum(model.losses),
            model.metrics[0].result(), model.metrics[1].result(), val_extra_loss))

  print("Training completed at: %s"%time.ctime())
  print(model.summary())

  #print(train_history.history)

  #At the end, save the full model
  #This is of limited usefulness due to difficulties saving and loading custom models
  #If can manage not to have any custom losses or metrics, loads just fine!
  #Well, fine except that it may lose the regularizer function
  #model.save(os.path.join(save_dir, 'model'), save_format='tf')


