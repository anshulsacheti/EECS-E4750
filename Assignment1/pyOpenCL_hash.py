import time
import argparse

import pyopencl as cl
import pyopencl.array

import numpy as np
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pdb

def setup_CL():
    """
    Sets up openCL platform devices,
        context, and CommandQueue

    Returns: list of device, context, CommandQueue
    """

    #Set up openCL platform
    NAME = 'NVIDIA CUDA'
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

    #Move data to device
    mf = cl.mem_flags
    name_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=name)
    b_dev = cl.Buffer(ctx, mf.WRITE_ONLY, name.nbytes)
    # name_dev = cl.array.to_device(queue, name)
    # b_dev = cl.array.empty(queue, name.shape, name.dtype)

    #Launch kernel
    #Only need global ID, no need for local
    prg = cl.Program(ctx, kernel).build()
    prg.func(queue, name.shape, None, name_dev, b_dev)

    #Save output
    hashed = np.empty_like(name)
    cl.enqueue_copy(queue, hashed, b_dev)

    print('input a: %s' % name)
    print('golden hash: %s' % [i % 17 for i in name])
    print('openCL hash: %s' % hashed)

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
    nameLength = []
    avgRunCount = 100
    for i in range(iterCount):

        #Scale length of name by iteration and convert to char
        name = np.array([ord(char) for char in refName]*(i+1)).astype(np.int32)

        #Move data to device
        mf = cl.mem_flags
        name_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=name)
        b_dev = cl.Buffer(ctx, mf.WRITE_ONLY, name.nbytes)
        # name_dev = cl.array.to_device(queue, name)
        # b_dev = cl.array.empty(queue, name.shape[0], name.dtype)

        #Launch kernel
        #Only need global ID, no need for local
        prg = cl.Program(ctx, kernel).build()


        #Run event and get avg run time
        tmp = []
        for j in range(avgRunCount):
            event = prg.func(queue, name.shape, None, name_dev, b_dev)
            event.wait()
            tmp.append(1e-9*(event.profile.end-event.profile.start))
        timeArray.append(np.average(tmp))

        # hashed = b_dev.get()
        hashed = np.empty_like(name)
        cl.enqueue_copy(queue, hashed, b_dev)
        nameLength.append(len(hashed))
        # print('input a: %s' % name)
        # print('golden hash: %s' % ([i % 17 for i in name]))
        # print('output hash: %s' % hashed)
        # print('-------------\n')

    print('opencl time:  %.15f' % np.average(timeArray))

    #Plot
    plt.gcf()
    ax = plt.figure().add_subplot(111)
    ax.plot(nameLength, timeArray)
    plt.xlabel('InputSize (number of chars)')
    plt.ylabel('RunTime (s)')
    plt.title('GPU openCL RunTime vs InputSize')
    plt.gca().set_xlim((min(nameLength), max(nameLength)))
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3e'))
    plt.savefig('GPU_openCL_plot.png')

def python_simple_hash(name):
    """
    Simple python hash
    Input:
        name: list of chars (string converted to list)
    Return: None
    Output: prints hash of character
    """

    hashed = np.zeros(len(name)).astype(int)
    count=0
    for char in name:
        hashed[count]=ord(char) % 17
        count+=1
    # print(hashed)

def python_multi_hash(name,iterCount):
    """
    MultiIter python hash
    Input:
        name: list of chars (string converted to list)
        iterCount: number of iterations in for loop
    Return: None
    Output: prints hash of character multiIter times
    """

    #Each iter start with N-character string and make it's length N*i
    #where i is the i-th iteration.
    timeArray = []
    nameLength = []
    refName = name
    for i in range(iterCount):
        hashed = np.zeros(len(refName)*(i+1)).astype(int)
        count=0
        name = refName*(i+1)
        start = time.time()
        for char in name:
            hashed[count]=ord(char) % 17
            count+=1
        timeArray.append(time.time()-start)
        nameLength.append(len(hashed))

    # print(hashed)
    print('python time:  %.15f' % np.average(timeArray))

    #Plot
    plt.gcf()
    ax = plt.figure().add_subplot(111)
    ax.plot(nameLength, timeArray)
    plt.xlabel('InputSize (number of chars)')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs InputSize")
    plt.gca().set_xlim((min(nameLength), max(nameLength)))
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3e'))
    plt.savefig('CPU_plot.png',bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name')
    parser.add_argument('--multiIter', type=int)

    args = parser.parse_args()
    if args.multiIter:
        multi_hash(list(args.name),args.multiIter)
        python_multi_hash(list(args.name), args.multiIter)
    else:
        simple_hash(list(args.name))
        python_simple_hash(list(args.name))
