# Libraries
import tensorflow as tf
import numpy as np
from models import VQVAETrainer

# Load dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5

data_variance = np.var(x_train / 255.0)

# Load VQ-VAE
vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=64)
# Compile it
vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam())
# Train the model
vqvae_trainer.fit(x=x_train_scaled, epochs=30, batch_size=128)
