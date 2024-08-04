
import tensorflow as tf
import numpy as np

#import os
#os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

from tensorflow import keras
from numba import cuda, float32

from model_pass_func import model_forward_pass_numpy, model_forward_pass_cuda

if __name__=="__main__":

    debug = False 

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (_,_), (test_images, test_labels) = mnist.load_data()

    # Load pre-trained model
    model = tf.keras.models.load_model('my_org_model_top4_quant.h5')

    # Get model weights and print model summary
    model_weights = model.get_weights()
    #model.summary()

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
        print("Shape of multipliers {}".format(np.shape(multipliers[0])))
    
        print ("Length of weights (", type(model_weights[0]), "): ", len(model_weights))
        print ("Weight shapes:")

        for w in model_weights:
            print(np.shape(w))


    for multiplier_type in range(0, 20):

        results = []
        for img in input_features:
            if debug: print ("Running pass of image {} with {} multiplier".format(img, multiplier_type))
            ref_result = model_forward_pass_numpy(img, model_weights, debug)

            print ("AFTER_NUMPY Length of weights (", type(model_weights[0]), "): ", len(model_weights))
            print ("Weight shapes:")

            for w in model_weights:
                print(np.shape(w))

            meas_result = model_forward_pass_cuda(img, model_weights, multipliers[multiplier_type], debug)
            
            print ("AFTER_CUDA Length of weights (", type(model_weights[0]), "): ", len(model_weights))
            print ("Weight shapes:")

            for w in model_weights:
                print(np.shape(w))

            #from pdb import set_trace; set_trace()

            #if debug:
            print("Measured array: {}, \ncorrect array: {}".format(meas_result, ref_result))
            print("Shapes MEAS: {}, \nCORRECT: {}".format(meas_result.shape, ref_result.shape))

            assert np.array_equal(ref_result, meas_result)
