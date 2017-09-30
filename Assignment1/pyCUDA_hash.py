import time
import argparse

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

import numpy as np
import string
import random
import os

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pdb

def simple_hash(name):
    """
    MultiIter CUDA hash
    Input:
        name: list of chars (string converted to list)
    Return: None
    Output: prints hash of characters
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
    start = driver.Event()
    end   = driver.Event()
    start.record()
    func(name_dev, b_dev, block = (name.shape[0],1,1))
    end.record()
    end.synchronize()
    tmp = 1e-3*start.time_till(end)

    # prg = cl.Program(ctx, kernel).build()
    # prg.func(queue, name.shape, None, name_dev, b_dev)

    #Save output
    hashed = b_dev.get()

    print('golden hash: %s' % [i % 17 for i in name])
    print('CUDA hash: %s' % hashed)
    print('CUDA single time:  %.15f' % tmp)

def multi_hash(name,iterCount):
    """
    MultiIter CUDA hash
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
    avgRunCount = 10
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
            print('CUDA multi hash: %s' % hashed)
        # print('-------------\n')

    print('golden hash: %s' % [i % 17 for i in name])
    print('CUDA multi hash: %s' % hashed)
    print('CUDA multi time:  %s' % timeArray)
    print('CUDA avg multi time: %.15f' % np.average(timeArray))

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
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('GPU_CUDA_plot.png')
    return timeArray

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
    start = time.time()
    for char in name:
        hashed[count]=ord(char) % 17
        count+=1
        end = time.time()-start

    print('python golden hash: %s' % hashed)
    print('python single runtime: %.15f' % end)

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

    print('python multi: %s' % hashed)
    print('python multi time:  %s' % timeArray)
    print('python avg multi time: %.15f' % np.average(timeArray))

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
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('CPU_plot.png',bbox_inches='tight')
    return timeArray

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name')
    parser.add_argument('--multiIter', type=int)

    args = parser.parse_args()
    if args.multiIter:
        tA_cuda=multi_hash(list(args.name),args.multiIter)
        tA_pyth=python_multi_hash(list(args.name), args.multiIter)

        for i in range(1,len(tA_cuda)):
            if tA_cuda[i-1]<=tA_pyth[i-1] and tA_cuda[i]<=tA_pyth[i]:
                print("CUDA was faster after the %d step" % (i*len(list(args.name))))
                break
    else:
        simple_hash(list(args.name))
        python_simple_hash(list(args.name))

#Referenced https://wiki.tiker.net/PyCuda/Examples/SimpleSpeedTest for timing
