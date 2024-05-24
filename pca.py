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

# Function to build and visualize PCA models
def build_pca_models(X_train,X_test, dims):
    mse_values = []
    for i, dim in enumerate(dims):

        # Fit PCA
        pca = PCA(n_components=dim)
        pca.fit(X_train)

        # Reconstruct images
        reconstructed_X = pca.inverse_transform(pca.transform(X_test))
        mse = np.mean((X_test - reconstructed_X) ** 2)
        mse_values.append(mse)

        plt.figure(figsize=(10, 2))
        print("latent: "+str(dim))
        print(f"MSE : {mse}")

        l=1
        for i in random_array:
            plt.subplot(1, 5, l)
            plt.imshow(reconstructed_X[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title("Test Reconstructed")
            l=l+1
        plt.show()

    return mse_values

# Build and visualize PCA models
mse_values = build_pca_models(X_train ,X_test, latent_dims)
plt.plot(latent_dims, mse_values, marker='o')
plt.title('MSE vs Latent Variable Dimension')
plt.xlabel('Latent Variable Dimension')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()
