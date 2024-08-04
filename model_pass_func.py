
import numpy as np
from scipy.signal import convolve2d

from numba import cuda, float32

from cuda_functions import cuda_zero_initialize, cuda_convolve2d, cuda_mat_scalar_add, cuda_maximum_elementwise, cuda_max_reduce1d_runner, cuda_polish_activation
from misc_func import recalc_new_kern_params

def model_forward_pass_cuda(input_features, model_weights, mult_lookup, debug=False):

    #from pdb import set_trace; set_trace()
    #1. kernel parameters need to be in the powers of 2
    #2. when calling a kernel, the input parameters should not be indexed

    if debug:
        print("Before sending data to memory.")

    #h_conv1_output = np.empty((28,28,64),dtype=np.float32)
    d_mw      = cuda.to_device(model_weights[0])
    d_inp_img = cuda.to_device(np.floor(input_features / 2))
    d_mult    = cuda.to_device(mult_lookup)

    d_conv1_output = cuda.device_array((28, 28, 64), dtype=np.float32)
    d_relu1_output = cuda.device_array((28, 28, 64), dtype=np.float32)

    h_model_weights = d_mw.copy_to_host()

    print("h_model_weights:")
    for w in model_weights:
        print(np.shape(w))

    #if debug:
    #    print("Before memory allocation.")
    #
    #cuda_zero_initialize[(1,1,64), (32,32)](d_conv1_output)

    #cuda.synchronize()

    #cio2 = cuda.device_array((28,28,64), dtype=np.float32)

    #for i in range(64):
        #block_size = (16, 16)
        # Rounding up to the whole blocks
        #grid_size = ((28 + block_size[0] - 1) // block_size[0],
                     #(28 + block_size[1] - 1) // block_size[1])

    cuda_convolve2d[(2,2,64), (16,16,1)](d_inp_img, d_mw, d_conv1_output, d_mult)
    cuda.synchronize()

    #if debug:
    #    print("Convolution 1 finished!")

    h_conv1_output = d_conv1_output.copy_to_host()

    h_model_weights = d_mw.copy_to_host()

    print("h_model_weights:")
    for w in model_weights:
        print(np.shape(w))

    return h_conv1_output
    
    #for i in range(64):
    #cuda_mat_scalar_add[(1,1,64), (32,32)](d_conv1_output, d_mw[1], d_conv1_output)
    #cuda.synchronize()

    #cuda_maximum_elementwise[(1,1,64),(32,32)](d_conv1_output, 0, d_relu1_output)
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

def model_forward_pass_numpy(input_features, model_weights, debug=False):
    """
    Custom function to perform forward pass through the modified model.
    """
    # Perform a series of convolutions and activations
    conv1_input = np.floor(input_features / 2)
    conv1_input = conv1_input.reshape(28, 28, 1)
    conv1_output = np.zeros([28, 28, 64])

    if debug:
        print ("Length of weights (", type(model_weights[0]), "): ", len(model_weights))
        print ("Weight shapes:")

        for w in model_weights:
            print(np.shape(w))


    for i in range(64):
        for j in range(1):
            conv1_output[:, :, i] += convolve2d(conv1_input[:, :, j], np.flip(model_weights[0][:, :, j, i]), mode = "valid")
        #conv1_output[:, :, i] += model_weights[1][i]

    return conv1_output


    relu1_output = np.maximum(0, conv1_output)

    relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)
    
    #conv2_output = np.zeros([28, 28, 32])
    #for i in range(32):
    #    for j in range(64):
    #        conv2_output[:, :, i] += convolve2d(relu1_output[:, :, j], np.flip(model_weights[2][:, :, j, i]), mode="valid")
    #    conv2_output[:, :, i] += model_weights[3][i]
    #relu2_output = np.maximum(0, conv2_output)
    #relu2_output = np.round((relu2_output / np.max(relu2_output)) * 127)
    #
    #conv3_output = np.zeros([28, 28, 16])
    #for i in range(16):
    #    for j in range(32):
    #        conv3_output[:, :, i] += convolve2d(relu2_output[:, :, j], model_weights[4][:, :, j, i], mode="valid")
    #    conv3_output[:, :, i] += model_weights[5][i]
    #relu3_output = np.maximum(0, conv3_output)
    #relu3_output = np.round((relu3_output / np.max(relu3_output)) * 127)
    #
    #conv4_output = np.zeros([26, 26, 8])
    #for i in range(8):
    #    for j in range(16):
    #        conv4_output[:, :, i] += convolve2d(relu3_output[:, :, j], model_weights[6][:, :, j, i], mode="valid")
    #    conv4_output[:, :, i] += model_weights[7][i]
    #relu4_output = np.maximum(0, conv4_output)
    #relu4_output = np.round((relu4_output / np.max(relu4_output)) * 127)
    #
    #conv5_output = np.zeros([24, 24, 4])
    #for i in range(4):
    #    for j in range(8):
    #        conv5_output[:, :, i] += convolve2d(relu4_output[:, :, j], model_weights[8][:, :, j, i], mode="valid")
    #    conv5_output[:, :, i] += model_weights[9][i]
    #relu5_output = np.maximum(0, conv5_output)
    #relu5_output = np.round((relu5_output / np.max(relu5_output)) * 127)
    #
    #flatten_output = np.reshape(relu5_output, [1, 2304])
    #fc1_output = np.matmul(flatten_output, model_weights[10]) + model_weights[11]
    #relu6_output = np.maximum(0, fc1_output) + 0.000001
    #relu6_output = np.round((relu6_output / np.max(relu6_output)) * 127)
    #
    #fc2_output = np.matmul(relu6_output, model_weights[12]) + model_weights[13]
    #relu7_output = np.maximum(0, fc2_output) + 0.000001
    #relu7_output = np.round((relu7_output / np.max(relu7_output)) * 127)
    #
    #fc3_output = np.matmul(relu7_output, model_weights[14]) + model_weights[15] + 0.000001
    #fc3_output = np.round((fc3_output / np.max(fc3_output)) * 127)
    
    #return np.argmax(fc3_output)

