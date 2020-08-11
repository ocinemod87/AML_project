import tensorflow as tf
import numpy as np
from tqdm import tqdm


@tf.function
def maxout(x, W, b, output_units, inter_units):
    n = tf.matmul(x, W) + b
    n = tf.reshape(n, (-1,  output_units, inter_units))
    return tf.reduce_max(n, axis=2)

def maxout_CNN(x, K, height, width, output_units, inter_units):
    n = tf.nn.conv2d(x, K, 1,  padding='SAME')
    output_height = n.shape[1]
    output_width = n.shape[2]

    #print('INSIDE MAXOUT Before REDUCE MAX '+str(n.shape))
    n = tf.reshape(n, (-1,  height, width, output_units, inter_units))
    n = tf.reduce_max(n, axis=4)
    #print('INSIDE MAXOUT AFTER REDUCE MAX '+str(n.shape))
    return tf.nn.max_pool2d(n, 2, 2, padding='VALID')

def generate_kernel(filter_height, filter_width, in_channels, out_channels):
    initializer = tf.initializers.GlorotUniform()
    return tf.Variable(initializer(shape=(filter_height, filter_width, in_channels, out_channels), dtype=tf.float64))



def generate_weights(input_units, output_units):
    initializer = tf.initializers.GlorotUniform()
    W = tf.Variable(initializer(shape=(input_units, output_units), dtype=tf.float64))
    b = tf.Variable(tf.zeros((output_units), dtype=tf.float64))

    return W, b


def default_optimisers():
    return tf.optimizers.RMSprop(), tf.optimizers.RMSprop()
