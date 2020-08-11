import tensorflow as tf
import keras
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from datetime import datetime
now = datetime.now()

tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


def discriminator_loss(data_prediction, noise_prediction):
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  return bce(tf.ones_like(data_prediction), data_prediction) + bce(tf.zeros_like(noise_prediction), noise_prediction)
  #return -tf.reduce_mean(tf.math.log(data_prediction) + tf.math.log(1-noise_prediction))

def generator_loss(noise_prediction):
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  return bce(tf.ones_like(noise_prediction), noise_prediction)
  #return -tf.reduce_mean(tf.math.log(noise_prediction))

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs

# def maxout(x, W, b):
#  n = tf.matmul(x, W) + b
#  return tf.reduce_max(n, axis=2)


def generator(x, shape, reuse=False):
  with tf.variable_scope("generator") as scope:
    if reuse:
      scope.reuse_variables()
    W_1 = tf.get_variable('W1', [100, 1200], initializer = tf.random_normal_initializer(stddev=0.01))
    b_1 = tf.get_variable('b1', [1200], initializer = tf.constant_initializer(0))
    W_2 = tf.get_variable('W2', [1200, 1200], initializer = tf.random_normal_initializer(stddev=0.01))
    b_2 = tf.get_variable('b2', [1200], initializer = tf.constant_initializer(0))
    W_3 = tf.get_variable('W3', [1200, 784], initializer = tf.random_normal_initializer(stddev=0.01))
    b_3 = tf.get_variable('b3', [784], initializer = tf.constant_initializer(0))
    x = tf.reshape(x, shape)
    x = tf.nn.leaky_relu(tf.matmul(x, W_1) + b_1)
    x = tf.nn.leaky_relu(tf.matmul(x, W_2) + b_2)
    return tf.nn.tanh(tf.matmul(x, W_3) + b_3)


#m1 = keras.layers.core.MaxoutDense(240, 5)
#m2 = keras.layers.core.MaxoutDense(240, 5)


def discriminator(x, shape, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()
    W_1 = tf.get_variable('W1', [784, 240], initializer = tf.random_normal_initializer(stddev=0.01))
    b_1 = tf.get_variable('b1', [240], initializer = tf.constant_initializer(0))
    W_2 = tf.get_variable('W2', [240, 240], initializer = tf.random_normal_initializer(stddev=0.01))
    b_2 = tf.get_variable('b2', [240], initializer = tf.constant_initializer(0))
    W_3 = tf.get_variable('W3', [240, 1], initializer = tf.random_normal_initializer(stddev=0.01))
    b_3 = tf.get_variable('b3', [1], initializer = tf.constant_initializer(0))
    x = tf.reshape(x, shape)
    #x = maxout(x, W_1, b_1)
    #x = maxout(x, W_2, b_2)
    #x = m1(x)
    #x = m2(x)
    x = tf.nn.leaky_relu(tf.matmul(x, W_1) + b_1)
    x = tf.nn.leaky_relu(tf.matmul(x, W_2) + b_2)
    #x = max_out(tf.matmul(x, W_1) + b_1, 240)
    #x = max_out(tf.matmul(x, W_2) + b_2, 240)
    return tf.nn.sigmoid(tf.matmul(x, W_3) + b_3)

def generate_maxout_weights(input_units, output_units, n_units):
  initializer = tf.initializers.GlorotUniform()
  W = tf.Variable(initializer(shape=(input_units, n_units, output_units)))
  b = tf.Variable(tf.zeros((n_units, output_units)))

  return W, b

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

n_input = 28 * 28
n_noise = 100

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# gW_1, gB_1, gW_2, gB_2, gW_3, gB_3 = generator_weights()

# dW_1, dB_1, dW_2, dB_2, dW_3, dB_3 = discriminator_weights()
g_out = generator(Z,(-1, 100))

d_gen = discriminator(g_out, (-1, 784), reuse=False)
d_real = discriminator(X, (-1, 784), reuse=True)

# generator_weights = [gW_1, gW_2, gW_3, gB_1, gB_2, gB_3]
# discriminator_weights = [dW_1, dW_2, dW_3, dB_1, dB_2, dB_3]
D_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
G_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
#print('G_var_list:', len(G_var_list))
#print('D_var_list:', len(D_var_list))

learning_rate = 0.1

loss_g = generator_loss(d_gen)
loss_d = discriminator_loss(d_real, d_gen)

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_d, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_g, var_list=G_var_list)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  batch_size = 128
  # batched = dataset.batch(batch_size)
  # length = len(list(batched))
  epochs = 100
  k = 1
  sample_size = 10

  for e in range(epochs):
    print(f'\nEpoch: {e}')

    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      batch_x = (batch_x - 127.5) / 127.5
      noise = np.random.uniform(-1, 1, size=(batch_x.shape[0], 100)).astype(np.float32)
      # noise = get_noise(batch_size, n_noise)

      _, loss_val_D = sess.run([train_D, loss_d],
                                feed_dict={X: batch_x, Z: noise})
      _, loss_val_G = sess.run([train_G, loss_g],
                                feed_dict={Z: noise})


    print('Epoch:', '%04d' % epochs,
        'D loss: {:.4}'.format(loss_val_D),
        'G loss: {:.4}'.format(loss_val_G))
    test_noise = np.random.uniform(-1, 1, size=(10, 100)).astype(np.float32)
    samples = sess.run(g_out, feed_dict={Z: test_noise})

    fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

    for i in range(sample_size):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))
    plt.show()

    # plt.savefig('samples/{}.png'.format(str(epochs).zfill(3)), bbox_inches='tight')
    plt.close(fig)


# img = tf.reshape(generator_model(tf.random.normal((1, 100))), (28, 28))

# import matplotlib.pyplot as plt

# plt.imshow(img)

# img_first = img
# plt.imshow(img_first)

# feature_list = [i for i in dataset_train]

# features = feature_list[500]
# fake = generator_model(tf.random.normal((1, 100)))
# x = features["image"]
# x = tf.cast(x, tf.float32) / 255.0
# d = discriminator_model(fake)
# d

# x = tf.constant([0.5001, 2.5, 2.3, 1.5, -4.5])
# tf.round(x)
