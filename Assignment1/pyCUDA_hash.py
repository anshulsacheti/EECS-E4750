import time
import argparse

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

import numpy as np
import os

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pdb

def simple_hash(name):
    """
    MultiIter python hash
    Input:
        name: list of chars (string converted to list)
        iterCount: number of iterations in for loop
    Return: None
    Output: prints hash of character multiIter times
    """

    #Setup CUDA

    name = np.array([ord(char) for char in name]).astype(np.int32)

    #CUDA Kernel
    kernel = """
    __global__ void func(int* a, int* b) {
        const int i = threadIdx.x;
        b[i] = a[i] % 17;
    }
    """

    #Move data to device
    name_dev = gpuarray.to_gpu(name)
    b_dev    = gpuarray.empty(name.shape[0], np.int32)
    # name_dev = cl.array.to_device(queue, name)
    # b_dev = cl.array.empty(queue, name.shape, name.dtype)

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("func")

    #Launch kernel
    #Number of threads equal to size of name
    func(name_dev, b_dev, block = (name.shape[0],1,1))

    # prg = cl.Program(ctx, kernel).build()
    # prg.func(queue, name.shape, None, name_dev, b_dev)

    #Save output
    hashed = b_dev.get()

    print('input a: %s' % name_dev.get())
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

    #CUDA Kernel
    kernel = """
    __global__ void func(int* a, int* b) {
        unsigned int i = threadIdx.x;
        b[i] = a[i] % 17;
    }
    """

    #Each iter start with N-character string and make it's length N*i
    #where i is the i-th iteration.
    refName = name
    timeArray = []
    nameLength = []
    avgRunCount = 1000
    for i in range(iterCount):

        #Scale length of name by iteration and convert to char
        name = np.array([ord(char) for char in refName]*(i+1)).astype(np.int32)

        #Move data to device
        name_dev = gpuarray.to_gpu(name)
        b_dev    = gpuarray.empty(name.shape[0], np.int32)
        # name_dev = cl.array.to_device(queue, name)
        # b_dev = cl.array.empty(queue, name.shape[0], name.dtype)

        #Compile kernel
        compiled = compiler.SourceModule(kernel)

        #Get compiled kernel
        func = compiled.get_function("func")

        #Run event and get avg run time
        tmp = []

        for j in range(avgRunCount):

            #Launch kernel
            #Number of threads equal to size of name
            start = driver.Event()
            end   = driver.Event()
            start.record()
            func(name_dev, b_dev, block = (name.shape[0],1, 1))
            end.record()
            end.synchronize()
            tmp.append(1e-3*start.time_till(end))
        timeArray.append(np.average(tmp))

        #Save output
        hashed = b_dev.get()
        nameLength.append(len(hashed))

        #Printing for testing purposes
        comp = sum(hashed==[i % 17 for i in name])==hashed.shape[0]
        if not(comp):
            print('input a: %s' % name)
            print('golden hash: %s' % ([i % 17 for i in name]))
            print('output hash: %s' % hashed)
            print('len: %d, sum: %d' % (hashed.shape[0],sum(hashed==[i % 17 for i in name])))
        # print('-------------\n')

    print('cuda time:  %.15f' % np.average(timeArray))

    #Plot
    plt.gcf()
    ax = plt.figure().add_subplot(111)
    ax.plot(nameLength, timeArray)
    plt.xlabel('InputSize (number of chars)')
    plt.ylabel('RunTime (s)')
    plt.title('GPU CUDA RunTime vs InputSize')
    plt.gca().set_xlim((min(nameLength), max(nameLength)))
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3e'))
    plt.savefig('GPU_CUDA_plot.png')

import time
import argparse

import numpy as np

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pdb

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

#Referenced https://wiki.tiker.net/PyCuda/Examples/SimpleSpeedTest for timing
