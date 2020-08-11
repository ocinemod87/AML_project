# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from datetime import datetime
now = datetime.now()


dataset_train = tfds.load(name='mnist', split=tfds.Split.TRAIN)
dataset_test = tfds.load(name='mnist', split=tfds.Split.TEST)

writer = tf.summary.create_file_writer(
    f'/Users/Anton/Documents/git/AML-project/logs/l5/{now.strftime("%Y%m%d-%H%M%S")}')


def discriminator_loss(data_prediction, noise_prediction):
    return -tf.reduce_mean(tf.math.log(data_prediction) + tf.math.log(tf.ones_like(noise_prediction)-noise_prediction))


def generator_loss(noise_prediction):
    return tf.reduce_mean(tf.ones_like(noise_prediction)-tf.math.log(noise_prediction))


def train_step(generator,
               discriminator,
               generator_weights,
               discriminator_weights,
               generator_optimizer,
               discriminator_optimizer,
               image_batch):

    x = image_batch
    #x = (tf.cast(x, tf.float32) - 127.5) / 127.5
    x = (tf.cast(x, tf.float32)) / 255.0
    noise = np.random.uniform(-1, 1,
                              size=(x.shape[0], 100)).astype(np.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        data_prediction = discriminator(x)
        noise_prediction = discriminator(generator(noise))

        d_loss = discriminator_loss(data_prediction, noise_prediction)
        d_grad = disc_tape.gradient(d_loss, discriminator_weights)
        discriminator_optimizer.apply_gradients(
            zip(d_grad, discriminator_weights))

        g_loss = generator_loss(noise_prediction)
        g_grad = gen_tape.gradient(g_loss, generator_weights)
        generator_optimizer.apply_gradients(
            zip(g_grad, generator_weights))


def train(generator,
          discriminator,
          generator_weights,
          discriminator_weights,
          dataset,
          valset,
          generator_optimizer,
          discriminator_optimizer,
          epochs,
          k):

    metrics = {
        'generator': {
            'avg': [],
            'accuracy': []
        },
        'discriminator': {
            'accuracy': []
        }
    }

    batch_size = 128
    batched = dataset.batch(batch_size)
    length = len(list(batched))

    data_iter = iter(dataset)
    real = [next(data_iter)['image'] for _ in range(250)]
    #real = [(tf.cast(x, tf.float32) - 127.5) / 127.5 for x in real]
    real = [tf.cast(x, tf.float32) / 255.0 for x in real]

    i = 0
    for e in range(epochs):
        print(f'\nEpoch: {e}')

        # Validation for the generator
        noise = np.random.uniform(-1, 1, size=(250, 100)).astype(np.float32)
        print(noise[0, 0])
        fake = generator(noise)
        result = discriminator(fake)
        real_res = discriminator(real)

        avg = tf.reduce_mean(result)
        accuracy = tf.reduce_sum(tf.round(result))/250.0

        metrics['generator']['avg'].append(avg)
        metrics['generator']['accuracy'].append(accuracy)

        print(f'Average: {avg}, accuracy: {accuracy}')

        disc_acc_fake = 1-tf.reduce_sum(tf.round(result))/255.0
        disc_acc_real = tf.reduce_sum(tf.round(real_res))/255.0
        disc_acc = (disc_acc_fake+disc_acc_real)/2

        metrics['discriminator']['accuracy'].append((disc_acc))

        #fig, ax = plt.subplots(1, 10)
        with writer.as_default():
            tf.summary.scalar('Noiseval', noise[0, 0], step=e)
            tf.summary.scalar('Accuracy', accuracy, step=e)
            tf.summary.scalar('Average', avg, step=e)
            tf.summary.scalar('Discriminative accuracy', disc_acc, step=e)
            tf.summary.image('Noise image', tf.reshape(
                noise+0.5, (-1, 10, 10, 1)), step=e, max_outputs=10)
            tf.summary.image(f'Images from epoch', tf.reshape(
                fake, (-1, 28, 28, 1)), step=e, max_outputs=10)
        # plt.show()

        for features in tqdm(dataset.batch(batch_size), desc=f'Epoch: {e} :: ', total=length):
            x = features['image']

            train_step(generator,
                       discriminator,
                       generator_weights,
                       discriminator_weights,
                       generator_optimizer,
                       discriminator_optimizer,
                       x)

            with writer.as_default():
                tf.summary.scalar('Gradient for W0',
                                  generator_weights[0][0][0], step=i)
                i += 1

    return metrics


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


@tf.function
def maxout(x, W, b, output_units, inter_units):
  n = tf.matmul(x, W) + b
  n = tf.reshape(n, (-1,  output_units, inter_units))
  return tf.reduce_max(n, axis=2)


def gen_generator(shape, W_1, W_2, W_3, b_1, b_2, b_3):
    @tf.function
    def generator(x):
        x = tf.reshape(x, shape)
        x = tf.nn.relu(tf.matmul(x, W_1) + b_1)
        x = tf.nn.relu(tf.matmul(x, W_2) + b_2)
        return tf.nn.sigmoid(tf.matmul(x, W_3) + b_3)
    return generator

#m1 = keras.layers.core.MaxoutDense(240, 5)
#m2 = keras.layers.core.MaxoutDense(240, 5)


def gen_discriminator(shape, W_1, W_2, W_3, b_1, b_2, b_3):
    @tf.function
    def discriminator(x):
        x = tf.reshape(x, shape)
        #x = maxout(x, W_1, b_1)
        #x = maxout(x, W_2, b_2)
        x = maxout(x, W_1, b_1, 240, 5)
        x = maxout(x, W_2, b_2, 240, 5)
        #x = m2(x)
        #x = tf.nn.leaky_relu(tf.matmul(x, W_1) + b_1)
        #x = tf.nn.leaky_relu(tf.matmul(x, W_2) + b_2)
        #x = max_out(tf.matmul(x, W_1) + b_1, 240)
        #x = max_out(tf.matmul(x, W_2) + b_2, 240)
        return tf.nn.sigmoid(tf.matmul(x, W_3) + b_3)
    return discriminator


def generate_weights(input_units, output_units):
    initializer = tf.initializers.GlorotUniform()
    W = tf.Variable(initializer(shape=(input_units, output_units)))
    b = tf.Variable(tf.zeros((output_units)))

    return W, b


def generate_maxout_weights(input_units, output_units, n_units):
    initializer = tf.initializers.GlorotUniform()
    W = tf.Variable(initializer(shape=(input_units, output_units*n_units)))
    b = tf.Variable(tf.zeros((output_units*n_units)))

    return W, b


gW_1, gB_1 = generate_weights(100, 1200)
gW_2, gB_2 = generate_weights(1200, 1200)
gW_3, gB_3 = generate_weights(1200, 784)

dW_1, dB_1 = generate_weights(784, 240*5)
dW_2, dB_2 = generate_weights(240, 240*5)
dW_3, dB_3 = generate_weights(240, 1)

generator_model = gen_generator((-1, 100), gW_1, gW_2, gW_3, gB_1, gB_2, gB_3)
discriminator_model = gen_discriminator(
    (-1, 784), dW_1, dW_2, dW_3, dB_1, dB_2, dB_3)

generator_weights = [gW_1, gW_2, gW_3, gB_1, gB_2, gB_3]
discriminator_weights = [dW_1, dW_2, dW_3, dB_1, dB_2, dB_3]

train(generator_model,
      discriminator_model,
      generator_weights,
      discriminator_weights,
      dataset_train,
      dataset_test,
      tf.optimizers.RMSprop(),
      tf.optimizers.RMSprop(),
      #tf.optimizers.SGD(learning_rate=0.01, momentum=0.05),  # Generator
      #tf.optimizers.SGD(learning_rate=0.01, momentum=0.05),  # Discriminator
      500,
      1)

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
