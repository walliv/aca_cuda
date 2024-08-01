
import numpy as np

from numba import cuda
from scipy.signal import convolve2d

from cuda_functions import cuda_convolve2d

def conv_kernel_test(mult_lookup):

    rng = np.random.default_rng()
    img_sizes = rng.integers(low=15, high = 256, size=100)
    kern_sizes = np.arange(2, 8)

    # Initialize kernels with random values
    kernels = []
    for ks in kern_sizes:
        kernels.append(rng.integers(low=1, high = 20, size=(ks,ks)))

    images = []
    for ims in img_sizes:
        images.append(rng.integers(low=20, high=256, size=(ims,ims)))

    d_mult_lookup = cuda.to_device(mult_lookup)

    for img in images:
        for kern in kernels:

            ref_image = convolve2d(img, kern, mode='valid') 

            h_res = np.zeros(img.shape)

            flip_kern = np.ascontiguousarray(np.flip(kern))
            #flip_kern = kern

            d_img = cuda.to_device(img)
            d_kern = cuda.to_device(flip_kern)
            d_res = cuda.device_array((img.shape[0] - kern.shape[0] + 1, img.shape[1] - kern.shape[1] + 1))

            #threads_x = int(np.abs(img.shape[0] - kern.shape[0]) + 1)
            #threads_y = int(np.abs(img.shape[1] - kern.shape[1]) + 1)
            block_size = (16, 16)
            # Rounding up to the whole blocks
            grid_size = ((d_res.shape[0] + block_size[0] - 1) // block_size[0],
                         (d_res.shape[1] + block_size[1] - 1) // block_size[1])

            print(f"Calling kernel on image of size {img.shape} and kernel of size {kern.shape} with grid size {grid_size} and block size {block_size}.")
            cuda_convolve2d[grid_size, block_size](d_img, d_kern, d_res, d_mult_lookup)
 
            #print("Calling kernel {} on image {} with threads_x: {} and threads_y: {}".format(kern, img, threads_x, threads_y))
            #cuda_convolve2d[1, (128, 128)](d_img, d_kern, d_res, d_mult_lookup)
        
            h_res = d_res.copy_to_host()
            
            print("Orig image shape: ", img.shape)
            print("Reference: ", ref_image, ", shape: ", ref_image.shape)
            print("Computed: ", h_res, ", shape: ", h_res.shape)
    
            assert np.allclose(h_res,ref_image, atol=1e-6)
    


if __name__=="__main__":
    # Load pre-computed multiplier arrays
    multipliers = [np.load(f"S_{i}.npy") for i in range(41, 45)] + \
                [np.load(f"S_{i}.npy") for i in range(51, 55)] + \
                [np.load(f"S_{i}.npy") for i in range(61, 65)] + \
                [np.load(f"S_{i}.npy") for i in range(71, 75)] + \
                [np.load(f"S_{i}.npy") for i in range(81, 85)]
 
    conv_kernel_test(multipliers[0])