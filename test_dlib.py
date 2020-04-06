
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

np.random.seed(42)
tf.random.set_seed(130692)

from libVAE import dataloaders

dspritesData = tfds.load("dsprites", split="train[:10]")
dataset = np.array([tf.cast(dat['image'], 'float32').numpy() for dat in dspritesData])
datatf = tf.convert_to_tensor(dataset[:10], dtype=float)
fakedata = tf.random.uniform((10,64,64,1), minval=0.0, maxval=1.0, seed=42)
fakemeans = tf.random.uniform((10,10), minval=-1.0, maxval=1.0, seed=42)
fakelogvars = tf.math.log(tf.random.uniform((10,10), minval=0.5, maxval=1.0, seed=42))
dataones = tf.ones((1,64,64,1))
datazeros = tf.zeros((1,64,64,1))
otherrandom = np.random.random((10,10))
dataDict = {'real/fake':[datatf, fakedata], 'zeros/zeros':[datazeros, datazeros],
            'ones/zeros':[dataones, datazeros], 'ones/ones':[dataones, dataones]}
latentDict = {'fake':[fakemeans, fakelogvars], 'zeros/zeros':[np.zeros([10,10])]*2,
              'ones/zeros':[np.ones([10,10]), np.zeros([10,10])],
              'ones/ones':[np.ones([10,10])]*2}

from libVAE import losses

for key, val in dataDict.items():
  print('\n')
  bloss = losses.bernoulli_loss(val[0], val[1], activation='logits')
  print("Bernoulli loss with sigmoid on %s data:"%(key))
  print(bloss)
  bloss_tanh = losses.bernoulli_loss(val[0], val[1], activation='tanh')
  print("Bernoulli loss with tanh on %s data:"%(key))
  print(bloss_tanh)
  l2loss = losses.l2_loss(val[0], val[1], activation='logits')
  print("L2 (squared) loss with sigmoid on %s data:"%(key))
  print(l2loss)
  l2loss_tanh = losses.l2_loss(val[0], val[1], activation='tanh')
  print("L2 (squared) loss with tanh on %s data:"%(key))
  print(l2loss_tanh)
  reconloss = losses.reconstruction_loss()
  print("Default reconstruction loss function on %s:"%(key))
  print(reconloss(val[0], val[1]))
  bin_entropy_loss = tf.keras.losses.binary_crossentropy(val[0], val[1], from_logits=True)
  bin_entropy_loss = tf.reduce_sum(bin_entropy_loss, axis=[1,2])
  print("Reduced binary entropy loss from keras on %s:"%(key))
  print(bin_entropy_loss)
  mse_loss = tf.keras.losses.mse(val[0], tf.math.sigmoid(val[1]))
  mse_loss = tf.reduce_sum(mse_loss, axis=[1,2])
  print("Reduced L2 loss from keras on %s:"%(key))
  print(mse_loss)

print('\n')
for key, val in latentDict.items():
  kldiv = losses.compute_gaussian_kl(val[0], val[1])
  print("KL divergence for %s data: %f"%(key, kldiv))

from libVAE import vae

print("\nTesting compute_covariance_z_mean")
testCovList = [(True, 0.0, 1.0), (True, 0.0, 4.0), (True, 1.0, 1.0),
               (False, np.zeros(10), np.ones([10,10])),
               (False, np.zeros(10), 0.5*(otherrandom+otherrandom.T)+np.diag(np.ones(10))*10)]
for vals in testCovList:
  if vals[0]:
    samples = tf.random.normal(shape=(100000, 10), mean=vals[1], stddev=tf.math.sqrt(vals[2]))
    cov = np.diag(np.ones([10])) * vals[2]
  else:
    samples = tf.constant(np.random.multivariate_normal(vals[1], vals[2], size=(1000000)))
    cov = vals[2]
  print(np.sum((vae.compute_covariance_z_mean(samples)-cov)**2))

print("\nTesting regularize_diag_off_diag_dip")
testDiagList = [np.ones([10,10]), np.zeros([10,10]),
                np.diag(np.ones(10)), 2.0*np.diag(np.ones(10))]
for val in testDiagList:
    tfval = tf.convert_to_tensor(val, dtype=np.float32)
    print(vae.regularize_diag_off_diag_dip(tfval, 1, 1))

print("\nTesting gaussian_log_density")
testLogDensList = [0.0, 1.0]
for val in testLogDensList:
  matrix = tf.ones(1)
  print(vae.gaussian_log_density(matrix, val, 0.0)[0])

print("\nTesting total_correlation")
testTotList = [1, 10]
for val in testTotList:
  z = tf.random.normal(shape=(10000, val))
  z_mean = tf.zeros(shape=(10000, val))
  z_logvar = tf.zeros(shape=(10000, val))
  print(vae.total_correlation(z, z_mean, z_logvar))

from libVAE import architectures

testencoder = architectures.ConvEncoder(10, kernel_initializer=tf.constant_initializer(value=0.5))
print(testencoder(dataones))
testdecoder = architectures.DeconvDecoder((64,64,1), kernel_initializer=tf.constant_initializer(value=0.5))
print(testdecoder(tf.ones((1,10), dtype='float32')))

#allmodels = [vae.BaseVAE((64,64,1), 10),
#             vae.BetaVAE((64,64,1), 10, beta=384.0), 
#             vae.DIPVAE((64,64,1), 10),
#             vae.BetaTCVAE((64,64,1), 10, beta=384.0)]
#
#from libVAE import train
#print("\n\n")
#
#for vaemodel in allmodels[2:3]:
#  print("\n")
#  print(vaemodel.name)
#  train.train(vaemodel, '', #May provide test file in future
#              save_dir='results_%s'%vaemodel.name)
#  this_eval = vaemodel.evaluate(datatf, datatf)
#
#
