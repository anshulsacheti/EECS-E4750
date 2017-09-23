import time
import argparse

import pyopencl as cl
import pyopencl.array

import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import pdb

def setup_CL():
    """
    Sets up openCL platform devices,
        context, and CommandQueue

    Returns: list of device, context, CommandQueue
    """

    #Set up openCL platform
    NAME = 'Apple'
    platforms = cl.get_platforms()
    dev = None
    for p in platforms:
        if p.name == NAME:
            dev = p.get_devices()

    # Command queue, enable GPU profiling
    ctx = cl.Context(dev)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

    return [dev,ctx,queue]

def simple_hash(name):
    """
    MultiIter python hash
    Input:
        name: list of chars (string converted to list)
        iterCount: number of iterations in for loop
    Return: None
    Output: prints hash of character multiIter times
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    name = np.array([ord(char) for char in name]).astype(np.int32)

    #openCL Kernel
    kernel = """
    __kernel void func(__global int* a, __global int* b) {
        unsigned int i = get_global_id(0);
        b[i] = a[i] % 17;
    }
    """

    #Move data to cpu
    name_dev = cl.array.to_device(queue, name)
    b_dev = cl.array.empty(queue, name.shape, name.dtype)

    #Launch kernel
    #Only need global ID, no need for local
    prg = cl.Program(ctx, kernel).build()
    prg.func(queue, name.shape, None, name_dev.data, b_dev.data)

    hashed = b_dev.get()

    print('input a: %s' % name)
    print('golden hash: %s' % [i % 17 for i in name])
    print('output hash: %s' % hashed)

def multi_hash(name,iterCount):
    """
    MultiIter python hash
    Input:
        name: list of chars (string converted to list)
        iterCount: number of iterations in for loop
    Return: None
    Output: prints hash of character multiIter times
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    #openCL Kernel
    kernel = """
    __kernel void func(__global int* a, __global int* b) {
        unsigned int i = get_global_id(0);
        b[i] = a[i] % 17;
    }
    """

    #Each iter start with N-character string and make it's length N*i
    #where i is the i-th iteration.
    refName = name
    timeArray = []
    for i in range(iterCount):

        #Scale length of name by iteration and convert to char
        name = np.array([ord(char) for char in refName]*(i+1)).astype(np.int32)

        #Move data to cpu
        name_dev = cl.array.to_device(queue, name)
        b_dev = cl.array.empty(queue, name.shape[0], name.dtype)

        #Launch kernel
        #Only need global ID, no need for local
        prg = cl.Program(ctx, kernel).build()

        event = prg.func(queue, name.shape, None, name_dev.data, b_dev.data)
        event.wait()
        timeArray.append(1e-9*(event.profile.end-event.profile.start))
        hashed = b_dev.get()

        # print('input a: %s' % name)
        # print('golden hash: %s' % ([i % 17 for i in name]))
        # print('output hash: %s' % hashed)
        # print('-------------\n')

    print('opencl time:  %.15f' % np.average(timeArray))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name')
    parser.add_argument('--multiIter', type=int)

    args = parser.parse_args()
    if args.multiIter:
        multi_hash((args.name),args.multiIter)
    else:
        simple_hash(list(args.name))


# M = 3
# times = []
# for i in range(M):
#     start = time.time()
#     a+b
#     times.append(time.time()-start)
# print('python time:  %f' % np.average(times))
#
# times = []
# for i in range(M):
#     evt = prg.func(queue, a.shape, None, a_gpu.data, b_gpu.data, c_gpu.data)
#     evt.wait()
#     times.append(1e-9*(evt.profile.end-evt.profile.start))
# print('opencl time:  %f' % np.average(times))
