
import numpy as np
from numba import cuda, float32
from misc_func import recalc_new_kern_params

@cuda.jit
def cuda_convolve2d(img, kern, res, mult):
    x, y, z = cuda.grid(3)

    # TODO: Copy the kernel to the shared memory
    if x < (img.shape[0] - kern.shape[0] + 1) and y < (img.shape[1] - kern.shape[1] + 1) and z < kern.shape[3]:

        #if i == 0 and j == 0:
        #    from pdb import set_trace; set_trace()

        tmp = 0.0

        for kr in range(kern.shape[0]):
            for kc in range(kern.shape[1]):
                #tmp += mult[int(kern[k,l])+128, int(img[i + k, j + l])+128]

                tmp = tmp + (img[x+kr, y+kc] * kern[kr,kc,0,z])

        res[x,y,z] = tmp

@cuda.jit
def cuda_maximum_elementwise(inp_arr, scalar, res):
    i, j, k = cuda.grid(3)

    if i < inp_arr.shape[0] and j < inp_arr.shape[1] and k < inp_arr.shape[2]:
        tmp = inp_arr[i,j,k]

        res[i,j,k] = max(tmp,scalar);

@cuda.jit
def cuda_max_reduce1d(inp_data, res):
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

    return d_partial_sums

@cuda.jit
def cuda_polish_activation(arr, maxval):
    i, j, k = cuda.grid(3)

    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        read_val = arr[i,j,k]
        arr[i,j,k] = round((read_val / maxval) * 127)

@cuda.jit
def cuda_zero_initialize(res):
    i, j, k = cuda.grid(3)

    if i < res.shape[0] and j < res.shape[1] and k < res.shape[2]: 
        res[i,j, k] = 0.0
    
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
            #tmp += mult[int(a[i, k])+128, int(b[k, j])+128]
            tmp += a[i, k] * b[k, j]

        res[i, j] = tmp

@cuda.jit
def cuda_mat_add(a, b, c):
    i, j = cuda.grid(2)

    if i < c.shape[0] and j < c.shape[1]:
        c[i,j] = a[i,j] + b[i,j]

@cuda.jit
def cuda_mat_scalar_add(arr, scalar, res):
    i, j, k = cuda.grid(3)

    if i < arr.shape[0] and j < arr.shape[1] and k < arr.shape[2]:
        res[i,j,k] = arr[i,j,k] + scalar[k] 
