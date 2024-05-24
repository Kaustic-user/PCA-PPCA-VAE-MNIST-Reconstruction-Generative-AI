import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_probability as tfp
from IPython import display

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images to 2D arrays
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
print(X_train.shape)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(X_test.shape)

random_array = np.random.randint(0, 9999, 5)

# Define the dimensions of latent variables
latent_dims = [2, 4, 8, 16, 32, 64]

#plotting 5 original test cases
plt.figure(figsize=(10, 2))
l=1
for i in random_array:
    plt.subplot(1, 5, l)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title('Test Original')
    l=l+1
plt.show()

train_dataset = X_train.reshape((X_train.shape[0], 28, 28, 1))
test_dataset = X_test.reshape((X_test.shape[0], 28, 28, 1))

batchsize = 40
train_size=60000
test_size=10000
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
epochs=20

train_images = (tf.data.Dataset.from_tensor_slices(train_dataset)
                 .shuffle(train_size).batch(batchsize))
test_images = (tf.data.Dataset.from_tensor_slices(test_dataset)
                .shuffle(test_size).batch(batchsize))

class VAE(tf.keras.Model):

  def __init__(self,latent):
    super(VAE, self).__init__()
    self.latent = latent
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(latent + latent),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent,)),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(784),
            tf.keras.layers.Reshape(target_shape=(28,28,1)),
        ]
    )


  def training(self, z=None):
    if z is None:
      z = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(z, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def decode(self, z, apply_sigmoid=False):
    gen = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(gen)
      return probs
    return gen


def train_step(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  grad = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grad, model.trainable_variables))


def compute_loss(model, x):
  mean, log_var = model.encode(x)
  z = tf.random.normal(shape=mean.shape) * tf.exp(log_var * .5) + mean
  x_generated = model.decode(z)
  crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_generated, labels=x)
  lpxz = -tf.reduce_sum(crossentropy, axis=[1, 2, 3]) #reconstruction error
  lpz = prob_dist_log_normal(z, 0., 0.)   #KL
  lqzx = prob_dist_log_normal(z, mean, log_var)
  return -tf.reduce_mean(lpxz + lpz - lqzx)


def prob_dist_log_normal(sample, mean, log_var, raxis=1):
  lp = tf.math.log(2. * np.pi)
  ans=tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-log_var) + log_var + lp),axis=raxis)
  return ans



all_predictions = []
mse_errors = []
for latent_dim in latent_dims:
    model = VAE(latent_dim)

    # Training the model
    for epoch in range(1, epochs + 1):
        print("epoch : ", epoch)
        for train_x in train_images:
            train_step(model, train_x, optimizer)

    # Generate predictions for random_images
    mean, logvar = model.encode(test_dataset)
    z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * .5) + mean
    predictions = model.training(z)

    # Reshape predictions to match the shape of random_images
    predictions = tf.reshape(predictions, test_dataset.shape)

    mse = tf.reduce_mean(tf.square(test_dataset - predictions)).numpy()
    mse_errors.append(mse)
    all_predictions.append(predictions)


for i, predictions in enumerate(all_predictions):
    latent_dim = latent_dims[i]
    plt.figure(figsize=(10, 2))
    l = 1
    for j in random_array:
        plt.subplot(1, 5, l)
        plt.imshow(predictions[j, :, :], cmap='gray')
        plt.axis('off')
        plt.title("Test Reconstructed")
        l = l+1
    plt.tight_layout()
    plt.show()

for i in range(0,6):
  print(f"Latent dim = {latent_dims[i]} , MSE : {mse_errors[i]}")


# Plot the line graph for MSE vs. Latent Dimension
plt.plot(latent_dims, mse_errors, marker='o')
plt.title('Mean Squared Error vs. Latent Dimension')
plt.xlabel('Latent Dimension')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
