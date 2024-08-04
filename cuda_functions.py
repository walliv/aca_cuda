
import numpy as np
from numba import cuda, float32
from misc_func import recalc_new_kern_params

@cuda.jit
def cuda_convolve2d(img, kern, res, layer_width, mult):
    """
    Konvolution kernel that processes the 3D structure.
    """

    x, y, z = cuda.grid(3)

    if x < (img.shape[0] - kern.shape[0] + 1) and y < (img.shape[1] - kern.shape[1] + 1) and z < kern.shape[3]:

        tmp = 0.0 

        for kr in range(kern.shape[0]):
            for kc in range(kern.shape[1]):
                #tmp += mult[int(kern[kr,kc,z,layer_width])+128, int(img[x + kr, y + kc, z])+128]

                # Elementwise multiplication between the subset of an input
                # image and a kernel.  Notice that kernels are contained in the
                # 4D array. That is because of the dimensionality of the input
                # weights.
                tmp = tmp + (img[x+kr, y+kc, z] * kern[kr,kc,z, layer_width])

        res[x,y,layer_width] = tmp

@cuda.jit
def cuda_maximum_elementwise_3d(inp_arr, scalar, res):
    """
    An element-wise maximum between a 3D structure and a scalar.
    """
    i, j, k = cuda.grid(3)

    if i < inp_arr.shape[0] and j < inp_arr.shape[1] and k < inp_arr.shape[2]:
        tmp = inp_arr[i,j,k]

        res[i,j,k] = max(tmp,scalar);

@cuda.jit
def cuda_maximum_elementwise_1d(inp_arr, scalar, res):
    """
    An element-wise maximum between a vector and a scalar.
    """
    i = cuda.grid(1)

    if i < inp_arr.shape[0]:
        tmp = inp_arr[i][0]

        res[i] = max(tmp,scalar);

@cuda.jit
def cuda_max_reduce1d(inp_data, res):
    """
    A reduction of a vector to find a maximum element.
    """
    
    # Define shared memory
    sdata = cuda.shared.array(shape=0, dtype=float32)
    
    # Calculate thread ID and data index
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * (cuda.blockDim.x * 2) + cuda.threadIdx.x
    
    # Load input into shared memory
    mySum = 0.0
    if i < inp_data.shape[0]:
        mySum = inp_data[i]

    if i + cuda.blockDim.x < inp_data.shape[0]:
        mySum = max(mySum, inp_data[i + cuda.blockDim.x])
    
    sdata[tid] = mySum
    cuda.syncthreads()
    
    # Perform reduction in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:

        if tid < s:
            sdata[tid] = max(sdata[tid],sdata[tid + s])

        s = s // 2

        cuda.syncthreads()
    
    # Write result for this block to global memory
    if tid == 0:
        res[cuda.blockIdx.x] = sdata[0]

def cuda_max_reduce1d_runner(d_array, bpg, tpb, ref_res, debug=False):
    """
    A runner for the cuda_max_reduce1d. This was created to handle the specific
    behavior of the reduction operation on the GPU. The underlying kernel has
    to be called multiple times with different parameters. The input array is
    flattened because of the currently implemented kernel.
    """
    d_inp_mat = d_array.ravel()
    d_partial_sums = cuda.device_array(bpg, dtype=np.float32)

    h_partial_sums = np.zeros(bpg)

    while (d_inp_mat.shape[0] > tpb):

        if debug: 
            print("Kernel launch config: BPG {}, TPB: {} ".format(bpg,tpb))

        cuda_max_reduce1d[bpg,tpb,0,tpb](d_inp_mat,d_partial_sums)

        if debug:
            h_partial_sums = d_partial_sums.copy_to_host().astype(int)
            print("Intermediate result {} of len {}".format(h_partial_sums, len(h_partial_sums)))
            print("The correct value is in the array: {}".format(ref_res in h_partial_sums))

        bpg, tpb = recalc_new_kern_params(bpg)
        d_inp_mat = d_partial_sums
        d_partial_sums = d_partial_sums[:bpg]

    return d_partial_sums[0]

@cuda.jit
def cuda_polish_activation(arr, maxval):
    """
    Kernel that normalizes the activation function on the end of every layer.
    This operates on a 3D array that is suitable for the convolutional layers.
    """
    i, j, k = cuda.grid(3)

    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        read_val = arr[i,j,k]
        arr[i,j,k] = round((read_val / maxval) * 127)

@cuda.jit
def cuda_polish_activation_1d(arr, maxval):
    """
    Kernel that performs the normalization of the activation function.
    """
    i = cuda.grid(1)

    if i < arr.shape[0]:
        read_val = arr[i][0]
        arr[i] = round((read_val / maxval) * 127)

@cuda.jit
def cuda_vect_add(a,b,res):
    """
    Kernel to add two 1D vectors together.
    """
    tid =cuda.grid(1);

    if (tid < a.size and tid < b.size):
        res[tid] = a[tid][0] + b[tid]

@cuda.jit
def cuda_matmul(a, b, res, mult):
    """
    Classical matrix multiplication
    """
    i, j = cuda.grid(2)

    if i < res.shape[0] and j < res.shape[1]:
        tmp = 0.0
        for k in range(res.shape[1]):
            #tmp += mult[int(a[i, k])+128, int(b[k, j])+128]
            tmp += a[i, k] * b[k, j]

        res[i, j] = tmp

@cuda.jit
def cuda_vect_matmul(a, b, res, mult):
    """
    This if for the multiplication of the row vector and the matrix.
    """
    i, j = cuda.grid(2)

    if i < res.shape[0] and j < res.shape[1]:
        tmp = 0.0
        for k in range(res.shape[1]):
            #tmp += mult[int(a[k])+128, int(b[k, j])+128]
            tmp += a[k] * b[k, j]

        res[i, j] = tmp

@cuda.jit
def cuda_mat_scalar_add(arr, scalar, res):
    """
    This kernel operates on a 3D array and accepts an array of scalars where
    every element gest add to each layer of the array.
    """
    i, j, k = cuda.grid(3)

    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        res[i,j,k] = arr[i,j,k] + scalar[k] 

@cuda.jit
def cuda_mat_scalar_1d(arr, scalar, res):
    """
    Kernel that adds scalar to a vector.
    """
    i = cuda.grid(1)

    if i < arr.shape[0]:
        res[i] = arr[i][0] + scalar 
