"""
In this example, we will develop a Vector Quantized Variational Autoencoder (VQ-VAE).
VQ-VAE was proposed in Neural Discrete Representation Learning by van der Oord et al.


VQ-VAE has three main parts:
1. In contrast with traditional VAE, that uses continuous distribution 
   to represent the latent space. VQ-VAEs, on the other hand, operate on a discrete latent space,
   making the optimization problem simpler. It does so by maintaining a discrete codebook. 

2. Discrete codebook, developed by discretizing the distance between the encoder outpouts and the codebook embeddings

3. These discrete code words are then fed to the decoder, which is trained to generate reconstructed samples.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # This parameter is best kept between [0.25, 2] as per the paper.
        # Commitement loss is used to prevent the codebook to grow to fast as it is dimensionless.
        # It ensures that the encoder outputs commits to the codebook by adding a loss term between the l2 norm distance between them
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()

        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Since the volume of the embedding space is diamensionless, it can grow arbitrarily,
        # if the embeddings dot not train as fast the encoder parameters. To make sure the encoder
        # commits to an embedding and its output does not grow, the commitement loss is used
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )

        # The Codebook Embeddings are trained based on the follwing loss term
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


if __name__ == "__main__":
    vqvae = VectorQuantizer(embedding_dim=3, num_embeddings=30)
    input = tf.random.uniform(shape=(2, 2, 3), dtype=tf.float32)
    encoding_indices = vqvae.get_code_indices(tf.reshape(input, [-1, 3]))
    quantized = vqvae.call(input)
    import pdb

    pdb.set_trace()
