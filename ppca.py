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

class PPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None
        self.sigma2 = None
        self.mean = None

    def fit(self, X, max_iter=200, tol=1e-4):
        N, D = X.shape
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        self.W = np.random.randn(D, self.n_components)
        self.sigma2 = np.random.rand()

        for _ in range(max_iter):
            M = np.dot(self.W.T, self.W) + self.sigma2 * np.eye(self.n_components)
            M_inv = np.linalg.inv(M)
            Z = np.dot(X_centered, np.dot(self.W, M_inv))
            X_reconstructed = np.dot(Z, self.W.T) + self.mean
            diff = X_centered - X_reconstructed
            self.W = np.dot(np.dot(X_centered.T, Z), np.linalg.inv(np.dot(Z.T, Z) + N * self.sigma2 * M_inv))
            self.sigma2 = np.sum(np.square(diff)) / (N * D)

            if np.sum(np.square(diff)) < tol:
                break

    def transform(self, X):
        X_centered = X - self.mean
        M_inv = np.linalg.inv(np.dot(self.W.T, self.W) + self.sigma2 * np.eye(self.n_components))
        Z = np.dot(X_centered, np.dot(self.W, M_inv))
        return Z

    def inverse_transform(self, X_transformed):
        X_reconstructed = np.dot(X_transformed, self.W.T) + self.mean
        return X_reconstructed


ppca_models = {}

for dim in latent_dims:
    ppca_models[dim] = PPCA(n_components=dim)

# Fit PPCA models
for dim, model in ppca_models.items():
    model.fit(X_train)

mean_squared_errors = {}

for dim, model in ppca_models.items():
    reconstructed_images = model.inverse_transform(model.transform(X_test))
    mse = np.mean((X_test - reconstructed_images) ** 2)
    mean_squared_errors[dim] = mse
    print(f"MSE : {mse}")
    print(f"Latent Dim: {dim}")
    plt.figure(figsize=(10, 2))
    l=1
    for i in random_array:
        plt.subplot(1, 5,l)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Test Reconstructed')
        l=l+1
    plt.show()


plt.figure(figsize=(8, 6))
plt.plot(list(mean_squared_errors.keys()), list(mean_squared_errors.values()), marker='o', linestyle='-')
plt.title('Mean Squared Error vs. Latent Dimension')
plt.xlabel('Latent Dimension')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
