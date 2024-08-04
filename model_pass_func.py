
import numpy as np
import math
from scipy.signal import convolve2d

from numba import cuda

from cuda_functions import cuda_convolve2d, cuda_mat_scalar_add, cuda_maximum_elementwise_3d, cuda_maximum_elementwise_1d,cuda_max_reduce1d_runner, cuda_polish_activation, cuda_vect_matmul, cuda_vect_add, cuda_mat_scalar_1d, cuda_polish_activation_1d
from misc_func import recalc_new_kern_params

def cuda_conv_layer(d_inp_arr, h_model_weights, d_mult, layer_width, layer_depth, layer_idx):
    """
    CUDA accelerated convolutional layer. The data are all processed on the
    GPU, onle the weights are copied on every iteration.
    """

    # Create arrays for the layer outputs
    d_conv_output = cuda.device_array((d_inp_arr.shape[0], d_inp_arr.shape[0], layer_width), dtype=np.float32)
    d_relu_output = cuda.device_array((d_inp_arr.shape[0], d_inp_arr.shape[0], layer_width), dtype=np.float32)

    # The weights are copied to device separately for every convolutional
    # layer. This is to avoid the conflicts of indexing DeviceNDArrays a NumPy
    # arrays. The first array is used as kernels for convolution.  The second
    # array contains matrices that are added to each convolution iteration over
    # a specific layer.
    d_mw_fst      = cuda.to_device(h_model_weights[layer_idx*2])
    d_mw_snd      = cuda.to_device(h_model_weights[(layer_idx*2)+1])

    # The section of convolutions (applies even of weights)
    for lw in range(layer_width):
        cuda_convolve2d[(2,2,layer_depth), (16,16,1)](d_inp_arr, d_mw_fst, d_conv_output, lw, d_mult)

    # Apply odd weights to the convoluted layers
    cuda_mat_scalar_add[(1,1,layer_width), (32,32)](d_conv_output, d_mw_snd, d_conv_output)

    # Calculates elementwise maximum over the computed 3D sets of arrays (it gets rid of
    # negative values)
    cuda_maximum_elementwise_3d[(1,1,layer_width),(32,32)](d_conv_output, 0, d_relu_output)

    # A CUDA call of the np.max() equivalent which does a reduction over all elements to find
    # the maximum element in a set.
    max_of_maxes = cuda.device_array(1, dtype=np.float32)
    bpg, tpb = recalc_new_kern_params(d_relu_output.shape[0]*d_relu_output.shape[1]*d_relu_output.shape[2])
    max_of_maxes = cuda_max_reduce1d_runner(d_relu_output.ravel(), bpg, tpb, -1)
        
    # This is a final normalization of an activated layer, an equivalent to:
    #   relu_output = np.round((relu_output / np.max(relu_output)) * 127)
    #
    cuda_polish_activation[(1,1,layer_width), (32,32)](d_relu_output, max_of_maxes)

    return d_relu_output

def model_forward_pass_cuda(input_features, model_weights, mult_lookup):
    """
    The CUDA accelerated pass through a LeNet5 CNN. The input_features is a 2D
    image.  The model weights is a set of arrays that get add selectively to
    each array.  The mult_lookup is a custom lookup table to perform the
    multiplication of two integers.
    """

    # Import image (only 1 at a time) and the multiplier look-up table
    d_inp_img = cuda.to_device(np.reshape(np.floor(input_features / 2), (input_features.shape[0], input_features.shape[1], 1)))
    d_mult    = cuda.to_device(mult_lookup)

    # Pass through the convolutional layers
    d_relu1_output = cuda_conv_layer(d_inp_img, model_weights, d_mult, 64, 1, 0)
    d_relu2_output = cuda_conv_layer(d_relu1_output, model_weights, d_mult, 32, 64, 1)
    d_relu3_output = cuda_conv_layer(d_relu2_output, model_weights, d_mult, 16, 32, 2)
    d_relu4_output = cuda_conv_layer(d_relu3_output, model_weights, d_mult, 8, 16, 3)
    d_relu5_output = cuda_conv_layer(d_relu4_output, model_weights, d_mult, 4, 8, 4)
    
    # ================================================================================
    # Fully connected layer 1
    # ================================================================================
    d_relu6_output = cuda.device_array((1,128), dtype=np.float32)
    d_fc1_output = cuda.device_array((1,128), dtype=np.float32)
    d_mw_fst = cuda.to_device(model_weights[10])
    d_mw_snd = cuda.to_device(model_weights[11])

    tpb = (16,16) 
    bpg_x = math.ceil(d_fc1_output.shape[0] / tpb[0])
    bpg_y = math.ceil(d_fc1_output.shape[1] / tpb[1])
    bpg = (bpg_x, bpg_y)
    cuda_vect_matmul[bpg, tpb](d_relu5_output.ravel(), d_mw_fst, d_fc1_output, d_mult)
    cuda_vect_add[1, 128](d_fc1_output, d_mw_snd, d_fc1_output)

    # Activation function 6
    cuda_maximum_elementwise_1d[1,128](d_fc1_output, 0.0, d_relu6_output)
    cuda_mat_scalar_1d[1,128](d_relu6_output, 0.000001, d_relu6_output)

    # Final polish of activation 6
    max_of_maxes = cuda.device_array(1, dtype=np.float32)
    bpg, tpb = recalc_new_kern_params(d_relu6_output.shape[0]*d_relu6_output.shape[1])
    max_of_maxes = cuda_max_reduce1d_runner(d_relu6_output, bpg, tpb, -1)
    cuda_polish_activation_1d[1,128](d_relu6_output, max_of_maxes)

    # ================================================================================
    # Fully connected layer 2
    # ================================================================================
    d_fc2_output = cuda.device_array((1,64), dtype=np.float32)
    d_relu7_output = cuda.device_array((1,64), dtype=np.float32)
    d_mw_fst = cuda.to_device(model_weights[12])
    d_mw_snd = cuda.to_device(model_weights[13])

    tpb = (16,16) 
    bpg_x = math.ceil(d_fc2_output.shape[0] / tpb[0])
    bpg_y = math.ceil(d_fc2_output.shape[1] / tpb[1])
    bpg = (bpg_x, bpg_y)
    cuda_vect_matmul[bpg, tpb](d_relu6_output, d_mw_fst, d_fc2_output, d_mult)
    cuda_vect_add[1, 64](d_fc2_output, d_mw_snd, d_fc2_output)

    # Activation function 7
    cuda_maximum_elementwise_1d[1,64](d_fc2_output, 0.0, d_relu7_output)
    cuda_mat_scalar_1d[1,64](d_relu7_output, 0.000001, d_relu7_output)

    # Final polish of activation 7
    max_of_maxes = cuda.device_array(1, dtype=np.float32)
    bpg, tpb = recalc_new_kern_params(d_relu7_output.shape[0]*d_relu7_output.shape[1])
    max_of_maxes = cuda_max_reduce1d_runner(d_relu7_output, bpg, tpb, -1)
    cuda_polish_activation_1d[1,64](d_relu7_output, max_of_maxes)

    # ================================================================================
    # Fully connected layer 3
    # ================================================================================
    d_fc3_output = cuda.device_array((1,10), dtype=np.float32)
    d_relu8_output = cuda.device_array((1,10), dtype=np.float32)
    d_mw_fst = cuda.to_device(model_weights[14])
    d_mw_snd = cuda.to_device(model_weights[15])

    tpb = (16,16) 
    bpg_x = math.ceil(d_fc3_output.shape[0] / tpb[0])
    bpg_y = math.ceil(d_fc3_output.shape[1] / tpb[1])
    bpg = (bpg_x, bpg_y)
    cuda_vect_matmul[bpg, tpb](d_relu7_output, d_mw_fst, d_fc3_output, d_mult)
    cuda_vect_add[1, 16](d_fc3_output, d_mw_snd, d_fc3_output)
    cuda_mat_scalar_1d[1,16](d_fc3_output, 0.000001, d_fc3_output)

    # Activation function 8
    max_of_maxes = cuda.device_array(1, dtype=np.float32)
    bpg, tpb = recalc_new_kern_params(d_relu8_output.shape[0]*d_relu8_output.shape[1])
    max_of_maxes = cuda_max_reduce1d_runner(d_relu8_output, bpg, tpb, -1)
    cuda_polish_activation_1d[1,16](d_relu8_output, max_of_maxes)
    
    return np.argmax(d_relu8_output.copy_to_host())

def model_forward_pass_numpy(input_features, model_weights):
    """
    A NumPy implementation of the LeNet5 neural network that gets passed sets
    of weights and one image. It serves as a reference implementation for
    testing.
    """
    conv1_input = np.floor(input_features / 2)
    conv1_input = conv1_input.reshape(28, 28, 1)
    conv1_output = np.zeros([28, 28, 64])

    for i in range(64): # width
        for j in range(1): # depth
            conv1_output[:, :, i] += convolve2d(conv1_input[:, :, j], np.flip(model_weights[0][:, :, j, i]), mode = "valid")
        conv1_output[:, :, i] += model_weights[1][i]

    relu1_output = np.maximum(0, conv1_output)
    relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)

    conv2_output = np.zeros([28, 28, 32])
    for i in range(1): # width 
        for j in range(64): # depth
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

