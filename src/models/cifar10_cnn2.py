from typing import *
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from util import maxout, maxout_CNN, generate_weights, generate_kernel


class Generator(tf.Module):

    def __init__(self, shape, output_shape):
        #print('shape inside init' + str(shape))
        self.shape = shape
        self.output_shape = output_shape
        self.W_1, self.b_1 = generate_weights(100, 8000)
        self.W_2, self.b_2 = generate_weights(8000, 8000)
        self.K = generate_kernel(5, 5, 3, 80)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def __call__(self, x):
        print('at the call beginning')
        #tf.print(x, output_stream=sys.stderr)
        x = tf.reshape(x, self.shape)

        x = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
        x = tf.nn.sigmoid(tf.matmul(x, self.W_2) + self.b_2)

        x = tf.reshape(x, (-1, 10, 10, 80))
        #tf.print(x.shape, output_stream=sys.stderr)
        batch = tf.shape(x)
        tf.print(batch)
<<<<<<< HEAD
        return tf.nn.sigmoid(tf.nn.conv2d_transpose(x, self.K, (batch[0], *self.output_shape), 3, padding='VALID'))
=======
        return tf.sigmoid(tf.nn.conv2d_transpose(x, self.K, (batch[0], *self.output_shape), 3, padding='VALID'))
>>>>>>> c9f96fa19aa2fc15bb1e5c0378abab3112434497


class Discriminator(tf.Module):

    def __init__(self, shape, maxout_units):
        self.shape = shape
        self.maxout_units = maxout_units
        self.K_1 = generate_kernel(8, 8, 3, 32*2)
        self.K_2 = generate_kernel(8, 8, 32, 32*2)
        self.K_3 = generate_kernel(5, 5, 32, 192*2)
        self.W_4, self.b_4 = generate_weights(
            3072, 500 * maxout_units)  # we need to debug this!!!!!!!
        self.W_5, self.b_5 = generate_weights(500, 1)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)])
    def __call__(self, x):

        #x = maxout_CNN(x, self.K_1, 32, 32, 32, 2)
        #x = maxout_CNN(x, self.K_2, 16, 16, 32, 2)
        #x = maxout_CNN(x, self.K_3, 8, 8, 192, 2)
        x = tf.nn.conv2d()

        #tf.print(x.shape, output_stream=sys.stderr)

        x = tf.reshape(x, (-1, 3072))

        x = maxout(x, self.W_4, self.b_4, 500, self.maxout_units)
        x = tf.matmul(x, self.W_5) + self.b_5
        #print('HERE IS THE SHAPE '+str(x.shape))
        return x


def default_models():
    generator_model = Generator((-1, 100), (32, 32, 3))
    discriminator_model = Discriminator((-1, (32, 32)), 5)

    return generator_model, discriminator_model


def default_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_train: tf.data.Dataset = tfds.load(
        name='cifar10', split=tfds.Split.TRAIN)
    dataset_test: tf.data.Dataset = tfds.load(
        name='cifar10', split=tfds.Split.TEST)

    return dataset_train, dataset_test
