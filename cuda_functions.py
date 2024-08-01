
import numpy as np
from numba import cuda, float32

@cuda.jit
def cuda_convolve2d(img, kern, res, mult):
    i, j = cuda.grid(2)

    # TODO: Copy the kernel to the shared memory
    if i < (img.shape[0] - kern.shape[0] + 1) and j < (img.shape[1] - kern.shape[1] + 1):
        tmp = 0

        for k in range(kern.shape[0]):
            for l in range(kern.shape[1]):
                #tmp += mult[int(kern[k,l])+128, int(img[i + k, j + l])+128]
                tmp += kern[k,l] * img[i+k, j+l]

        res[i,j] = tmp

@cuda.jit
def cuda_convolve_and_add(img, kern, res, mult):
    i, j = cuda.grid(2)

    # TODO: Copy the kernel to the shared memory
    if i < (img.shape[0] - kern.shape[0] + 1) and j < (img.shape[1] - kern.shape[1] + 1):
        tmp = res[i,j]

        for k in range(kern.shape[0]):
            for l in range(kern.shape[1]):
                #tmp += mult[int(kern[k,l])+128, int(img[i + k, j + l])+128]
                tmp += kern[k,l] * img[i+k, j+l]

        res[i,j] = tmp

@cuda.jit
def cuda_maximum_elementwise(scalar, inp_arr, res):
    i, j, k = cuda.grid(3)

    if i < inp_arr.shape[0] and j < inp_arr.shape[1] and k < inp_arr.shape[2]:
        tmp = inp_arr[i,j,k]

        res[i,j,k] = max(tmp,scalar);

@cuda.reduce
def cuda_max_reduce_simple(a,b):
    return max(a,b)

@cuda.jit
def cuda_max_reduce1d(inp_data, res):
    # Define shared memory
    sdata = cuda.shared.array(shape=0, dtype=cuda.float32)
    
    # Calculate thread ID and data index
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * (cuda.blockDim.x * 2) + cuda.threadIdx.x
    
    # Load input into shared memory
    mySum = 0.0
    if i < inp_data.shape[0]:
        mySum = inpdata[i]
    if i + cuda.blockDim.x < inp_data.shape[0]:
        mySum += inp_data[i + cuda.blockDim.x]
    
    sdata[tid] = mySum
    cuda.syncthreads()
    
    # Perform reduction in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:

        if tid < s:
            sdata[tid] += sdata[tid + s]

        s = s // 2

        cuda.syncthreads()
    
    # Write result for this block to global memory
    if tid == 0:
        res[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def cuda_max_reduce2d(array, res):
    # Define shared memory for block-wise reduction
    sdata = cuda.shared.array(shape=0, dtype=cuda.float32)
    
    # Calculate thread ID and data index for 2D array
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    i,j = cuda.grid(2)
    
    # Initialize local sum
    mySum = 0.0
    
    # Load input into shared memory if within bounds
    if i < array.shape[0] and j < array.shape[1]:
        mySum = array[i, j]
    
    # Store into shared memory
    sdata[ty * bw + tx] = mySum

    cuda.syncthreads()
    
    # Reduction in shared memory
    s = bw * bh // 2
    tid = ty * bw + tx
    
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        s //= 2
        cuda.syncthreads()
    
    # Write result for this block to global memory
    if tid == 0:
        res[by, bx] = sdata[0]

@cuda.jit
def cuda_zero_initialize(res):
    i, j, k = cuda.grid(3)

    if i < res.shape[0] and j < res.shape[1] and k < res.shape[2]:
        res[i,j,k] = 0
    
@cuda.jit
def cuda_vect_add(a,b,res):
    tid =cuda.grid(1);

    if (tid < a.size and tid < b.size):
        res[tid] = a[tid] + b[tid]

    
@cuda.jit
def cuda_vect_mult(a,b,res,mult):
    tid =cuda.grid(1)

    if (tid < a.size, tid < b.size):
        res[tid] = mult[int(a[tid])+128, int(b[tid])+128]

@cuda.jit
def cuda_matmul(a, b, res, mult):
    i, j = cuda.grid(2)
    if i < res.shape[0] and j < res.shape[1]:
        tmp = 0
        for k in range(a.shape[1]):
            tmp += mult[int(a[i, k])+128, int(b[k, j])+128]

        res[i, j] = tmp

@cuda.jit
def cuda_mat_add(a, b, c):
    i, j = cuda.grid(2)

    if i < c.shape[0] and j < c.shape[1]:
        c[i,j] = a[i,j] + b[i,j]