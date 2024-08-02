# -*- coding: utf-8 -*-
import time
import os
from numba import njit, cuda, float32, int32
import tensorflow as tf
import numpy as np
from tensorflow import keras

from cuda_functions import cuda_zero_initialize, cuda_convolve2d, cuda_mat_add, cuda_maximum_elementwise, cuda_max_reduce1d_runner, cuda_polish_activation
from misc_func import recalc_new_kern_params

def model_forward_pass_cuda(d_input_feature, d_model_weights, d_mult_lookup, debug=False):
    """
    Custom function to perform forward pass through the modified model.
    """
    d_conv1_output = cuda.device_array((28, 28, 64), dtype=np.float32)
    for i in range(64):
        cuda_zero_initialize[(1,1), (32,32)](d_conv1_output[:,:,i])
        cuda.synchronize()

    #cuda.synchronize()
    
    if debug:
        print("Initilize conv1_output to 0")
        print("Shape of the weights: ", d_model_weights.shape)

        for w in d_model_weights:
            print(w.shape)

    return d_conv1_output
    
    #cio1 = cuda.device_array((28,28,64), dtype=np.float32)
    #cio2 = cuda.device_array((28,28,64), dtype=np.float32)

    #for i in range(64):
    #    cuda_convolve2d[1, (32,32)](d_input_feature, d_model_weights[0][0][0][0][i], cio1[:,:,i], d_mult_lookup)

    #cuda.synchronize()

    #if debug:
    #    print("Convolution 1 finished!")
    #
    #for i in range(64):
    #    cuda_mat_add[1, (32,32)](cio1[:, :, i], d_model_weights[1][i], conv1_output[:, :, i])

    #cuda.synchronize()

    #relu1_output = cuda.device_array((28, 28, 64), dtype=np.float32)
    #cuda_maximum_elementwise[1,(28,28,64)](0, conv1_output, relu1_output)

    #cuda.synchronize()

    #max_of_maxes = cuda.device_array(1, dtype=np.float32)

    #bpg, tpb = recalc_new_kern_params(relu1_output.shape[0]*relu1_output.shape[1]*relu1_output.shape[2])
    #max_of_maxes = cuda_max_reduce1d_runner(relu1_output, bpg, tpb, -1)
    #    
    #cuda.synchronize()

    #cuda_polish_activation[1, (28,28,64)](relu1_output, max_of_maxes)

    #relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)
    #
    #conv2_output = cuda.device_array([28, 28, 32])
    #for i in range(32):
    #    for j in range(64):
    #        conv2_output[:, :, i] += custom_conv2d(relu1_output[:, :, j], np.flip(model_weights[2][:, :, j, i]), mult_lookup)
    #    conv2_output[:, :, i] += model_weights[3][i]
    #relu2_output = np.maximum(0, conv2_output)
    #relu2_output = np.round((relu2_output / np.max(relu2_output)) * 127)
    #
    #conv3_output = cuda.device_array([28, 28, 16])
#
    #for i in range(16):
    #    for j in range(32):
    #        conv3_output[:, :, i] += custom_conv2d(relu2_output[:, :, j], np.flip(model_weights[4][:, :, j, i]), mult_lookup)
    #    conv3_output[:, :, i] += model_weights[5][i]
    #relu3_output = np.maximum(0, conv3_output)
    #relu3_output = np.round((relu3_output / np.max(relu3_output)) * 127)
    #
    #conv4_output = cuda.device_array([26, 26, 8])
#
    #for i in range(8):
    #    for j in range(16):
    #        conv4_output[:, :, i] += custom_conv2d(relu3_output[:, :, j], np.flip(model_weights[6][:, :, j, i]), mult_lookup)
    #    conv4_output[:, :, i] += model_weights[7][i]
    #relu4_output = np.maximum(0, conv4_output)
    #relu4_output = np.round((relu4_output / np.max(relu4_output)) * 127)
    #
    #conv5_output = cuda.device_array([24, 24, 4])
#
    #for i in range(4):
    #    for j in range(8):
    #        conv5_output[:, :, i] += custom_conv2d(relu4_output[:, :, j], np.flip(model_weights[8][:, :, j, i]), mult_lookup)
    #    conv5_output[:, :, i] += model_weights[9][i]
    #relu5_output = np.maximum(0, conv5_output)
    #relu5_output = np.round((relu5_output / np.max(relu5_output)) * 127)
    #
    #flatten_output = np.reshape(relu5_output, [1, 2304])
    #fc1_output = np.empty()
    #cuda_matmul(flatten_output, model_weights[10], fc1_output, mult_lookup) 
    ## add fc1_output + model_weights[11]
    #relu6_output = np.maximum(0, fc1_output) + 0.000001
    #relu6_output = np.round((relu6_output / np.max(relu6_output)) * 127)
    #
    #fc2_output = cuda_matmul(relu6_output, model_weights[12], mult_lookup) + model_weights[13]
    #relu7_output = np.maximum(0, fc2_output) + 0.000001
    #relu7_output = np.round((relu7_output / np.max(relu7_output)) * 127)
    #
    #fc3_output = cuda_matmul(relu7_output, model_weights[14], mult_lookup) + model_weights[15] + 0.000001
    #fc3_output = np.round((fc3_output / np.max(fc3_output)) * 127)
    
    #return np.argmax(fc3_output)

if __name__=="__main__":

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
