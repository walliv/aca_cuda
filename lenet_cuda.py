# -*- coding: utf-8 -*-
import time
import os
from numba import njit, cuda, float32, int32
import tensorflow as tf
import numpy as np
from tensorflow import keras

from model_pass_func import model_forward_pass_cuda

if __name__=="__main__":

    debug = True

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (_,_), (test_images, test_labels) = mnist.load_data()

    # Load pre-trained model
    model = tf.keras.models.load_model('my_org_model_top4_quant.h5')

    # Get model weights and print model summary
    model_weights = model.get_weights()
    # model.summary()

    # use 1000 test image from mnist dataset
    batch_size = 10
    input_features = test_images[:batch_size]

    # Load pre-computed multiplier arrays
    multipliers = [np.load(f"S_{i}.npy") for i in range(41, 45)] + \
                [np.load(f"S_{i}.npy") for i in range(51, 55)] + \
                [np.load(f"S_{i}.npy") for i in range(61, 65)] + \
                [np.load(f"S_{i}.npy") for i in range(71, 75)] + \
                [np.load(f"S_{i}.npy") for i in range(81, 85)]

    if debug:
        print ("Length of weights (", type(model_weights[0]), "): ", len(model_weights))
        print(np.shape(multipliers[0]))
    
        for w in model_weights:
            print(np.shape(w))

    d_model_weights = cuda.to_device(model_weights)
    d_input_features = cuda.to_device(np.floor(input_features) / 2)

    for multiplier_type in range(0, 20):
        d_mult_lookup = cuda.to_device(multipliers[multiplier_type])

        # Record the start time
        start_time = time.time()

        results = []
        for image_index in range(1000):

            results.append(model_forward_pass_cuda(d_input_features[image_index], d_model_weights, d_mult_lookup))
            print(results)
            filename = f"Result_Approx_Multi_{multiplier_type}.npy"
            np.save(filename, results)
            print(results)
            print(np.size(results))

        elapsed_time = time.time() - start_time
        print('Execution time:', elapsed_time, 'seconds')

        # Calculate and print accuracy
        accuracy = np.sum(results == test_labels[:len(results)]) / len(results)
        print(f"Accuracy with multiplier {multiplier_type}: {accuracy}")
        print(f"Accuracy with multiplier {multiplier_type}: {accuracy}")
    # Loop through different multiplier types and calculate results
