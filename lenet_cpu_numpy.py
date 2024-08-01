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
from scipy.signal import convolve2d

    #exact_convol = signal.convolve2d(matr, kernel, mode = "same")

def model_forward_pass(input_features, model_weights, image_index):
    """
    Custom function to perform forward pass through the modified model.
    """
    # Perform a series of convolutions and activations
    conv1_input = np.floor(input_features[image_index] / 2)
    conv1_input = conv1_input.reshape(28, 28, 1)
    conv1_output = np.zeros([28, 28, 64])
    for i in range(64):
        for j in range(1):
            conv1_output[:, :, i] += convolve2d(conv1_input[:, :, j], np.flip(model_weights[0][:, :, j, i]), mode = "valid")
        conv1_output[:, :, i] += model_weights[1][i]
    relu1_output = np.maximum(0, conv1_output)
    relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)
    
    conv2_output = np.zeros([28, 28, 32])
    for i in range(32):
        for j in range(64):
            conv2_output[:, :, i] += convolve2d(relu1_output[:, :, j], np.flip(model_weights[2][:, :, j, i]), mode="valid")
        conv2_output[:, :, i] += model_weights[3][i]
    relu2_output = np.maximum(0, conv2_output)
    relu2_output = np.round((relu2_output / np.max(relu2_output)) * 127)
    
    conv3_output = np.zeros([28, 28, 16])
    for i in range(16):
        for j in range(32):
            conv3_output[:, :, i] += convolve2d(relu2_output[:, :, j], model_weights[4][:, :, j, i], mode="valid")
        conv3_output[:, :, i] += model_weights[5][i]
    relu3_output = np.maximum(0, conv3_output)
    relu3_output = np.round((relu3_output / np.max(relu3_output)) * 127)
    
    conv4_output = np.zeros([26, 26, 8])
    for i in range(8):
        for j in range(16):
            conv4_output[:, :, i] += convolve2d(relu3_output[:, :, j], model_weights[6][:, :, j, i], mode="valid")
        conv4_output[:, :, i] += model_weights[7][i]
    relu4_output = np.maximum(0, conv4_output)
    relu4_output = np.round((relu4_output / np.max(relu4_output)) * 127)
    
    conv5_output = np.zeros([24, 24, 4])
    for i in range(4):
        for j in range(8):
            conv5_output[:, :, i] += convolve2d(relu4_output[:, :, j], model_weights[8][:, :, j, i], mode="valid")
        conv5_output[:, :, i] += model_weights[9][i]
    relu5_output = np.maximum(0, conv5_output)
    relu5_output = np.round((relu5_output / np.max(relu5_output)) * 127)
    
    flatten_output = np.reshape(relu5_output, [1, 2304])
    fc1_output = np.matmul(flatten_output, model_weights[10]) + model_weights[11]
    relu6_output = np.maximum(0, fc1_output) + 0.000001
    relu6_output = np.round((relu6_output / np.max(relu6_output)) * 127)
    
    fc2_output = np.matmul(relu6_output, model_weights[12]) + model_weights[13]
    relu7_output = np.maximum(0, fc2_output) + 0.000001
    relu7_output = np.round((relu7_output / np.max(relu7_output)) * 127)
    
    fc3_output = np.matmul(relu7_output, model_weights[14]) + model_weights[15] + 0.000001
    fc3_output = np.round((fc3_output / np.max(fc3_output)) * 127)
    
    return np.argmax(fc3_output)

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
        results.append(model_forward_pass(input_features, model_weights, image_index))
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
