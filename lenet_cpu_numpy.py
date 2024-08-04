# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:10:40 2022

@author: sshakibhamedan
"""

import time
import os
import tensorflow as tf
import numpy as np

from tensorflow import keras

from model_pass_func import model_forward_pass_numpy

if __name__=="__main__":

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (_,_), (test_images, test_labels) = mnist.load_data()

    # Load pre-trained model
    model = tf.keras.models.load_model('my_org_model_top4_quant.h5')

    # Get model weights and print model summary
    model_weights = model.get_weights()
    model.summary()

    # use 1000 test image from mnist dataset
    batch_size = 1000
    input_features = test_images[:batch_size]

    # Record the start time
    start_time = time.time()

    results = []
    for image_index in range(1000):
        results.append(model_forward_pass_numpy(input_features[image_index], model_weights))
        print(results)
        filename = f"Result_Exact_Multi.npy"
        np.save(filename, results)
        print(results)
        print(np.size(results))

    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')

    # Calculate and print accuracy
    accuracy = np.sum(results == test_labels[:len(results)]) / len(results)
    print(f"Accuracy with NumPy multipliers: {accuracy}")
