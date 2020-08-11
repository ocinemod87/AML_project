from typing import *

import tensorflow as tf
import tensorflow_datasets as tfds

from util import maxout, generate_weights

class Generator(tf.Module):

    def __init__(self, shape):
        self.shape = shape
        self.W_1, self.b_1 = generate_weights(100, 8000)
        self.W_2, self.b_2 = generate_weights(8000, 8000)
        self.W_3, self.b_3 = generate_weights(8000, 3072)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def __call__(self, x):
        x = tf.reshape(x, self.shape)
        x = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
        x = tf.nn.relu(tf.matmul(x, self.W_2) + self.b_2)
        #return tf.nn.sigmoid(tf.matmul(x, self.W_3) + self.b_3)
        #x = tf.nn.sigmoid(tf.matmul(x, self.W_2) + self.b_2)
        return tf.nn.sigmoid(tf.matmul(x, self.W_3) + self.b_3)

class Discriminator(tf.Module):

    def __init__(self, shape, maxout_units):
        self.shape = shape
        self.maxout_units = maxout_units
        self.W_1, self.b_1 = generate_weights(3072, 1600 * maxout_units)
        self.W_2, self.b_2 = generate_weights(1600, 1600 * maxout_units)
        self.W_3, self.b_3 = generate_weights(1600, 1)

        self.parameters = [self.W_2, self.W_2, self.W_3, self.b_1, self.b_2, self.b_3]

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def __call__(self, x):
        x = tf.reshape(x, self.shape)

        x = maxout(x, self.W_1, self.b_1, 1600, self.maxout_units)
        x = maxout(x, self.W_2, self.b_2, 1600, self.maxout_units)

        return tf.matmul(x, self.W_3) + self.b_3


def default_models():
    generator_model = Generator((-1, 100))
    discriminator_model = Discriminator((-1, 3072), 2)

    return generator_model, discriminator_model


def default_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_train: tf.data.Dataset = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
    dataset_test: tf.data.Dataset = tfds.load(name='cifar10', split=tfds.Split.TEST)

    return dataset_train, dataset_test
