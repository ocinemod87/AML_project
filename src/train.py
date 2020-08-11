import os
import json
import argparse

import tensorflow as tf
import numpy as np

from datetime import datetime

from tqdm import tqdm

import importlib
import util

from models import mnist
from likelihood import likelihood

from models import cifar10_cnn
from models import cifar10_dense

def discriminator_loss(data_prediction, noise_prediction):
    return -tf.reduce_mean(tf.math.log(data_prediction) + tf.math.log(tf.ones_like(noise_prediction)-noise_prediction))


def generator_loss(noise_prediction):
    return tf.reduce_mean(tf.ones_like(noise_prediction)-tf.math.log(noise_prediction))


def train_step(generator,
               discriminator,
               generator_optimizer,
               discriminator_optimizer,
               image_batch):

    x = image_batch

    # Remember to normalise to correct value range based on the last activation function in the Generator
    #x = (tf.cast(x, tf.float64) - 127.5) / 127.5
    x = (tf.cast(x, tf.float64)) / 255.0
    noise = np.random.uniform(-1, 1,
                              size=(x.shape[0], 100)).astype(np.float64)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        data_prediction = discriminator(x)
        noise_prediction = discriminator(generator(noise))

        d_loss = discriminator_loss(data_prediction, noise_prediction)
        d_grad = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(d_grad, discriminator.trainable_variables))

        g_loss = generator_loss(noise_prediction)
        g_grad = gen_tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(g_grad, generator.trainable_variables))


def train(generator: tf.function,
          discriminator: tf.function,
          dataset,
          valset,
          generator_optimizer,
          discriminator_optimizer,
          epochs,
          writer,
          model_path):

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
    real = [next(data_iter)['image'] for _ in range(128)]
    # Remember to normalise to correct value range based on the last activation function in the Generator
    #real = [(tf.cast(x, tf.float64) - 127.5) / 127.5 for x in real]
    real = [tf.cast(x, tf.float64) / 255.0 for x in real]

    i = 0
    for e in range(epochs):
        print(f'\nEpoch: {e}')

        # Validation for the generator
        noise = np.random.uniform(-1, 1, size=(128, 100)).astype(np.float64)
        print(noise[0, 0])
        fake = generator(noise)
        result = discriminator(fake)
        real_res = discriminator(real)

        avg = tf.reduce_mean(result)
        accuracy = tf.reduce_sum(tf.round(result))/128

        metrics['generator']['avg'].append(avg)
        metrics['generator']['accuracy'].append(accuracy)

        print(f'Average: {avg}, accuracy: {accuracy}')

        disc_acc_fake = 1-tf.reduce_sum(tf.round(result))/128
        disc_acc_real = tf.reduce_sum(tf.round(real_res))/128
        disc_acc = (disc_acc_fake+disc_acc_real)/2

        metrics['discriminator']['accuracy'].append((disc_acc))

        #fig, ax = plt.subplots(1, 10)
        with writer.as_default():
            tf.summary.scalar('Noiseval', noise[0, 0], step=e)
            tf.summary.scalar('Accuracy', accuracy, step=e)
            tf.summary.scalar('Average', avg, step=e)
            tf.summary.scalar('Discriminative accuracy', disc_acc, step=e)


            tf.summary.image('Noise image', tf.reshape(
                noise, (-1, 10, 10, 1)), step=e, max_outputs=10)


            images = tf.reshape(fake[10:], (-1, *real[0].shape))
            images = tf.concat([img for img in images[:10]], axis=1)
            tf.summary.image('Combined image', tf.expand_dims(images, axis=0), step=e)
            tf.summary.image(f'Images from epoch', tf.reshape(
                fake, (-1, *real[0].shape)), step=e, max_outputs=10)

            mean, std = likelihood(dataset, valset, generator, val_size=1000)
            tf.summary.scalar('Likelihood mean', mean, step=e)
            tf.summary.scalar('Likelihood std', std, step=e)
        # plt.show()

        for features in tqdm(dataset.batch(batch_size), desc=f'Epoch: {e} :: ', total=length):
            x = features['image']
            print('SHAPE OF XXXXXXXX '+str(x.shape))
            train_step(generator,
                       discriminator,
                       generator_optimizer,
                       discriminator_optimizer,
                       x)

        tf.saved_model.save(generator, os.path.join(model_path, 'generator'))
        tf.saved_model.save(discriminator, os.path.join(model_path, 'discriminator'))

    return generator


def get_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', help='Path to config file')
    args = parser.parse_args()

    with open(args.conf) as f:
        return json.load(f)


if __name__ == '__main__':
    conf = get_conf()

    now = datetime.now()
    log_dir = os.path.join(conf['log_path'], conf['log_dir'], now.strftime("%Y%m%d-%H%M%S"))
    writer = tf.summary.create_file_writer(log_dir)

    try:
        module = importlib.import_module(f"models.{conf['model']}", 'src')

        model_args = module.default_models()
        print(model_args)
        dataset_args = module.default_datasets()
    except ModuleNotFoundError as e:
        raise ValueError(f'{conf["model"]} is not a valid model name! Must be one of the files in the models folder')

    if conf['optimiser'] == 'default':
        optimiser_args = util.default_optimisers()

    train(*model_args, *dataset_args, *optimiser_args, conf['epochs'], writer, conf['model_path'])
