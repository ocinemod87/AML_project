from typing import *
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

from util import maxout, maxout_CNN, generate_weights, generate_kernel

class Generator(tf.Module):

    def __init__(self, shape, output_shape):
        #print('shape inside init'+ str(shape))
        self.shape = shape
        self.output_shape = output_shape
        self.W_1, self.b_1 = generate_weights(100, 8000)
        self.W_2, self.b_2 = generate_weights(8000, 8000)
        self.K = generate_kernel(5, 5, 3, 80)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):
        #print('at the call beginning')
        #tf.print(x.shape, output_stream=sys.stderr)
        x = tf.reshape(x, self.shape)

        x = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
        x = tf.nn.sigmoid(tf.matmul(x, self.W_2) + self.b_2)
        #tf.print(x.shape)
        x = tf.reshape(x, (-1,10,10,80))
        #tf.print(x.shape, output_stream=sys.stderr)
        return tf.nn.conv2d_transpose(x, self.K, self.output_shape, 3, padding='VALID')

class Discriminator(tf.Module):

    def __init__(self, shape, maxout_units):
        self.shape = shape
        self.maxout_units = maxout_units
        self.K_1 = generate_kernel(8, 8, 3, 32*2)
        self.K_2 = generate_kernel(8, 8, 32, 32*2)
        self.K_3 = generate_kernel(5, 5, 32, 192*2)
        self.W_4, self.b_4 = generate_weights(3072, 500 * maxout_units) #we need to debug this!!!!!!!
        self.W_5, self.b_5 = generate_weights(500, 1)

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x):

        x = maxout_CNN(x, self.K_1, 32, 32, 32, 2)
        x = maxout_CNN(x, self.K_2, 16, 16, 32, 2)
        x = maxout_CNN(x, self.K_3, 8, 8, 192, 2)

        #tf.print(x.shape, output_stream=sys.stderr)

        x = tf.reshape(x, (-1, 3072))

        x = maxout(x, self.W_4, self.b_4, 500, self.maxout_units)
        x = tf.nn.sigmoid(tf.matmul(x, self.W_5) + self.b_5)
        #print('HERE IS THE SHAPE '+str(x.shape))
        return x


def default_models():
    generator_model = Generator((-1, 100), (128,124,124,3))
    discriminator_model = Discriminator((-1, (124,124)), 5)

    return generator_model, discriminator_model


def default_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def preprocessing(inputs):
        x = inputs['image']
        x = tf.image.crop_to_bounding_box(x,
                                        offset_height=0,
                                        offset_width=20,
                                        target_height=178,
                                        target_width=178)

        x = tf.image.resize_images(x, (124, 124))

        return {
            'image': x
        }

    dataset_train: tf.data.Dataset = tfds.load(name='celeb_a', split=tfds.Split.TRAIN).map(preprocessing)
    dataset_test: tf.data.Dataset = tfds.load(name='celeb_a', split=tfds.Split.TEST).map(preprocessing)

    return dataset_train, dataset_test


