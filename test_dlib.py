
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(130692)

from libVAE import dataloaders
dataset = dataloaders.raw_image_data('/home/local/NIST/jim2/CG_MC/Lattice_Gas/train_VAE/out2D.nc')
datatf = tf.convert_to_tensor(dataset[:10], dtype=float)
fakedata = tf.random.uniform((10,64,64,1), minval=0.0, maxval=1.0, seed=42)

from libVAE import losses
bloss = losses.bernoulli_loss(datatf, fakedata, activation='logits')
print(bloss)
bloss = losses.bernoulli_loss(datatf, fakedata, activation='logits', subtract_true_image_entropy=True)
print(bloss)
bloss_tanh = losses.bernoulli_loss(datatf, fakedata, activation='tanh')
print(bloss_tanh)
l2loss = losses.l2_loss(datatf, fakedata, activation='logits')
print(l2loss)
l2loss_tanh = losses.l2_loss(datatf, fakedata, activation='tanh')
print(l2loss_tanh)
fakemeans = tf.random.uniform((10,10), minval=-1.0, maxval=1.0, seed=42)
fakelogvars = tf.math.log(tf.random.uniform((10,10), minval=0.5, maxval=1.0, seed=42))
kldiv = losses.compute_gaussian_kl(fakemeans, fakelogvars)
print(kldiv)
reconloss = losses.reconstruction_loss()
print(reconloss(datatf, fakedata))
bin_entropy_loss = tf.keras.losses.binary_crossentropy(datatf, fakedata, from_logits=True)
print(bin_entropy_loss.shape)
bin_entropy_loss = tf.reduce_sum(bin_entropy_loss, axis=[1])
print(bin_entropy_loss)
mse_loss = tf.keras.losses.mse(datatf, tf.math.sigmoid(fakedata))
mse_loss = tf.reduce_sum(mse_loss, axis=[1])
print(mse_loss)

from libVAE import vae
testvae = vae.BaseVAE((64,64,1), 10)
(testencode) = testvae.encoder(datatf)
testsample = testvae.sampler(testencode[0], testencode[1])
testdecode = testvae.decoder(testsample)
testrecon = testvae(datatf)
#print(testencode)
#print(testsample)
#print(testdecode)
#print(testrecon)

allmodels = [testvae,
             vae.BetaVAE((64,64,1), 10, beta=384.0), 
             vae.DIPVAE((64,64,1), 10),
             vae.BetaTCVAE((64,64,1), 10, beta=384.0)]

from libVAE import train
print("\n\n")

for vaemodel in allmodels[2:3]:
  print("\n")
  print(vaemodel.name)
  train.train(vaemodel, '/home/local/NIST/jim2/CG_MC/Lattice_Gas/train_VAE/out2D.nc',
              save_dir='results_%s'%vaemodel.name)
  this_eval = vaemodel.evaluate(datatf, datatf)

