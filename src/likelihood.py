import tensorflow as tf
import numpy as np
from tqdm import tqdm

import argparse
import os

from models import mnist

import gc


@tf.function
def log_mean_exp(a):
    max_ = tf.reduce_max(a, axis=1)
    sum_a = tf.math.reduce_mean(tf.exp(a - tf.expand_dims(max_, axis=1)), axis=1)
    return max_ + tf.math.log(sum_a)


def gen_parzen(mu, sigma):
    @tf.function
    def parzen(x):
        # (x-mean)/h
        a = (tf.expand_dims(x, axis=1) - tf.expand_dims(mu, axis=0)) / sigma
        E = log_mean_exp(-0.5*tf.math.reduce_sum(tf.pow(a, 2), axis=2))
        Z = mu.shape[1] * tf.math.log(sigma * np.sqrt(np.pi*2))

        return E-Z
    return parzen


def get_nll(x, parzen, batch_size=32):

    nlls = []
    print(list(range(len(x)//batch_size)))
    for i in tqdm(range(len(x)//batch_size)):
        batch = x[i:i+batch_size, :]
        nll = parzen(batch)
        nlls.append(nll)

    return np.array(nlls)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        parzen = gen_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size)
        lls.append(tmp.mean())
        del parzen
        gc.collect()

    idx = np.argmax(lls)
    return sigmas[idx]


def likelihood(train_set, test_set, generator, val_size=1000):


    val_set = [tf.reshape(t['image'], (1, -1)) for t in tqdm(train_set)][:val_size]
    val_set = tf.concat(val_set, axis=0)
    val_set = tf.cast(val_set, tf.float64) / 255.0

    test_set = [tf.reshape(e['image'], (1, -1)) for e in tqdm(test_set)]
    size = 1000 #len(test_set)

    indices = np.random.randint(0, len(test_set), size=size)
    test_samples = [test_set[i] for i in indices]
    test_samples = tf.concat(test_samples, axis=0)
    test_samples = tf.cast(test_samples, tf.float64) / 255.0
    gc.collect()


    sigma_range = np.logspace(-1, 0, num=10)
    sigma = cross_validate_sigma(test_samples, val_set, sigma_range, batch_size=32)
    print(f'Sigma: {sigma}')

    print('generating samples')
    gen_inputs = np.random.uniform(-1, 1, size=(size, 100)).astype(np.float64)
    gen_samples = tf.cast(generator(gen_inputs), tf.float64)
    gen_samples = tf.reshape(gen_samples, (size, -1))
    parzen = gen_parzen(gen_samples, sigma)


    print('nll eval')
    print(test_samples.shape)
    result = get_nll(test_samples, parzen)
    print(f'Mean: {tf.reduce_mean(result)}, Std: {tf.math.reduce_std(result)/np.sqrt(len(test_samples))}')

    return tf.reduce_mean(result), tf.math.reduce_std(result)/np.sqrt(len(test_samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model path')
    parser.add_argument('dataset', help='Dataset')
    parser.add_argument('--val-size', default=1000)

    args = parser.parse_args()

    if args.dataset == 'mnist':
        train_set, test_set = mnist.default_datasets()
        val_set = [tf.reshape(t['image'], (1, -1)) for t in tqdm(train_set)][:args.val_size]
        val_set = tf.concat(val_set, axis=0)
        val_set = tf.cast(val_set, tf.float64) / 255.0

    generator = tf.saved_model.load(os.path.join(args.model, 'generator'))

    test_set = [tf.reshape(e['image'], (1, -1)) for e in tqdm(test_set)]
    size = 1000 #len(test_set)

    indices = np.random.randint(0, len(test_set), size=size)
    test_samples = [test_set[i] for i in indices]
    test_samples = tf.concat(test_samples, axis=0)
    test_samples = tf.cast(test_samples, tf.float64) / 255.0
    gc.collect()


    sigma_range = np.logspace(-1, 0, num=10)
    sigma = cross_validate_sigma(test_samples, val_set, sigma_range, batch_size=32)
    print(f'Sigma: {sigma}')



    print('generating samples')
    gen_inputs = np.random.uniform(-1, 1, size=(size, 100)).astype(np.float64)
    gen_samples = tf.cast(generator(gen_inputs), tf.float64)
    parzen = gen_parzen(gen_samples, sigma)


    print('nll eval')
    result = get_nll(test_samples, parzen)
    print(f'Mean: {tf.reduce_mean(result)}, Std: {tf.math.reduce_std(result)/np.sqrt(len(test_samples))}')



if __name__ == '__main__':
    main()