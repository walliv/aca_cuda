

def next_pow2(x):
    return 1 << (x - 1).bit_length()

def recalc_new_kern_params(size, max_thread_num=64):
    if size < max_thread_num * 2:
        tpb = next_pow2((size + 1) // 2)
    else:
        tpb = max_thread_num 

    bpg = (size + (tpb * 2 - 1)) // (tpb * 2)

    return bpg, tpb
