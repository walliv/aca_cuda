
import numpy as np
import math

from numba import cuda
from scipy.signal import convolve2d

from cuda_functions import cuda_convolve2d, cuda_matmul, cuda_max_reduce1d_runner
from misc_func import next_pow2, recalc_new_kern_params

def matmul_test(mult_lookup,debug=False):

    rng = np.random.default_rng()
    mat_sizes = rng.integers(low=15, high = 256, size=300)

    matrices = []
    for i in range(0,200,3):
        A = rng.integers(low=20, high=256, size=(mat_sizes[i],mat_sizes[i+1]))
        B = rng.integers(low=20, high=256, size=(mat_sizes[i+1],mat_sizes[i+2]))

        ref_result = np.matmul(A,B)

        d_matA = cuda.to_device(A)        
        d_matB = cuda.to_device(B)        
        d_mat_res = cuda.device_array((A.shape[0], B.shape[1]))
        h_mat_res = np.zeros((A.shape[0],B.shape[1]))

        tpb = (16,16) 
        bpg_x = math.ceil(h_mat_res.shape[0] / tpb[0])
        bpg_y = math.ceil(h_mat_res.shape[1] / tpb[1])
        bpg = (bpg_x, bpg_y)
        print("Running multiplication with {} TPB and {} BPG".format(tpb, bpg))

        cuda_matmul[bpg, tpb](d_matA, d_matB, d_mat_res, mult_lookup)

        h_mat_res = d_mat_res.copy_to_host()
        
        assert np.array_equal(ref_result, h_mat_res)    

def max_reduction_test(debug=True):
    no_of_tests = 500
    rng = np.random.default_rng()
    mat_sizes = rng.integers(low=15, high = 1024, size=no_of_tests)

    matrices = []
    for i in range(0,500,2):
        mat = rng.integers(low=20, high=256, size=(mat_sizes[i],mat_sizes[i+1]))

        print("No {}: Max reduction of a matrix with shape {}".format(i, mat.shape))
        ref_res = np.max(mat)
        flat_arr_len = mat.shape[0]*mat.shape[1]

        bpg, tpb = recalc_new_kern_params(flat_arr_len)

        d_mat = cuda.to_device(mat)
        #d_partial_sums = cuda.device_array(bpg, dtype=np.float32)
        d_res = cuda.device_array(1, dtype=np.float32)

        #h_partial_sums = np.zeros(bpg)
        h_res = np.zeros(1)

        #d_inp_mat = d_mat.ravel();

        #while (d_inp_mat.shape[0] > tpb):

        #    if debug: 
        #        print("Kernel launch config: BPG {}, TPB: {} ".format(bpg,tpb))

        #    cuda_max_reduce1d[bpg,tpb,0,tpb](d_inp_mat,d_partial_sums)

        #    if debug:
        #        h_partial_sums = d_partial_sums.copy_to_host().astype(int)
        #        print("Intermediate result {} of len {}".format(h_partial_sums, len(h_partial_sums)))
        #        print("The correct value is in the array: {}".format(ref_res in h_partial_sums))

        #    bpg, tpb = recalc_new_kern_params(bpg)
        #    d_inp_mat = d_partial_sums
        #    d_partial_sums = d_partial_sums[:bpg]

        d_partial_sums = cuda_max_reduce1d_runner(d_mat, bpg, tpb, ref_res, debug)

        #cuda_max_reduce1d[1,tpb,0,tpb](d_partial_sums,d_res)

        h_res = d_partial_sums.copy_to_host()

        if debug:
            print("Result of reduction {}".format(h_res))
            print("Correct result {}".format(ref_res))

        assert ref_res == h_res
        
def conv_kernel_test(mult_lookup, debug=False):

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
            
            if debug:
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
 
    print("===============================================================")
    print("Running test of convolution")
    print("===============================================================")
    conv_kernel_test(multipliers[0])

    #print("===============================================================")
    #print("Running test of matrix multiplication")
    #print("===============================================================")
    #matmul_test(multipliers[0])

    #print("===============================================================")
    #print("Running test of max reduction")
    #print("===============================================================")
    #max_reduction_test()
