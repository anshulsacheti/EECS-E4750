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

    #Ord(char) returns the ascii number for some character
    name = np.array([ord(char) for char in name]).astype(np.int32)


    #Setup CUDA
    #CUDA Kernel
    kernel = """
    __global__ void func(int* a, int* b) {
        int i = threadIdx.x;
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
    print('CUDA==golden: %s' % (sum(hashed==[i % 17 for i in name])==hashed.shape[0]))
    print('CUDA single time:  %.2E' % tmp)

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
        int i = threadIdx.x;
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
        #Ord(char) returns the ascii number for some character
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
            val = name.shape[0]
            start.record()
            func(name_dev, b_dev, block = (val,1, 1))
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
    print('CUDA==golden: %s' % (sum(hashed==[i % 17 for i in name])==hashed.shape[0]))
    print('CUDA multi time:  %s' % timeArray)
    print('CUDA avg multi time: %.2E' % np.average(timeArray))

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

def transpose_square_matrix(matrix):
    """
    Transpose square matrix via CUDA
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    kernel = """
    __global__ void func(float* a, float* b, const int matrix_row_size) {
        //unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
        unsigned int i = threadIdx.x;
        unsigned int j = threadIdx.y;
        b[i+(j+blockIdx.x)*blockDim.x]=a[(i+blockIdx.x)*blockDim.x+j];
        //b[i*blockDim.x+j]=a[i+j*blockDim.x];
        //b[i*blockDim.x+j] = threadIdx.x;
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix_float)
    transpose_gpu = gpuarray.empty(matrix.shape, np.float32)
    matrix_row_size = np.int32(matrix.shape[0])

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("func")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    threadCounts = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    #blockThreads = 8
    xThreads = min(int(matrix_row_size),4)
    yThreads = min(int(matrix_row_size),4)
    totalThreads = xThreads*yThreads
    blocks = np.int(max(np.ceil(matrix_val_count / totalThreads),1))

    print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transpose_gpu, matrix_row_size, block = (xThreads,yThreads,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposed = transpose_gpu.get()

    print('CUDA %d x %d transpose time:  %.2E' % (matrix.shape[0], matrix.shape[0], runtime))
    print('CUDA==golden: %s' % np.allclose(transposed, np.transpose(matrix)))
    if not(np.allclose(transposed, np.transpose(matrix))):
        # print('Original Matrix: %s' % matrix)
        print('golden transpose: %s' % np.transpose(matrix))
        print('CUDA val: %s' % transposed)
        print('CUDA transpose: %s' % np.isclose(np.transpose(matrix),transposed))
    print('--------------------')

    return [transposed, runtime]

def python_square_matrix_mult(matrix):
    """
    Calculate transpose of square matrix NxN
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    transposed_matrix = np.zeros([matrix.shape[0],matrix.shape[0]])
    start = time.time()
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[0]):
    #         transposed_matrix[i,j] = matrix[j,i]
    transposed_matrix = np.transpose(matrix)
    end = time.time()-start

    #Testing
    if not(np.allclose(transposed_matrix,np.transpose(matrix))):
        print(transposed_matrix)

    print('python transpose time:  %.2E' % end)
    return [transposed_matrix, end]

if __name__=="__main__":

    #Handle command line inputs
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("dim1", type=int)
    parser.add_argument("dim2", type=int, default=0)

    args = parser.parse_args()

    #initialize arrays
    cpu_transpose_array = []
    cpu_runtime_array = []

    gpu_transpose_array = []
    gpu_runtime_array = []

    dimSize = []

    #Handle square, nonsquare matrices
    if args.dim2!=0:
        print("Not developed")
    else:
        for i in range(1,5):
            #CPU Runtime
            tmp = np.random.rand(args.dim1*i, args.dim1*i)
            transposed, runtime = python_square_matrix_mult(tmp)
            cpu_transpose_array.append(transposed)
            cpu_runtime_array.append(runtime)

            #GPU Runtime
            transposed, runtime = transpose_square_matrix(tmp)
            gpu_transpose_array.append(transposed)
            gpu_runtime_array.append(runtime)

            dimSize.append(args.dim1*i)

    #Plot
    plt.gcf()
    # ax = plt.figure().add_subplot(111)
    # ax.plot(dimSize, gpu_runtime_array, 'r--', dimSize, cpu_runtime_array, 'g^')
    plt.plot(dimSize, gpu_runtime_array, 'r--', label="GPU")
    plt.plot(dimSize, cpu_runtime_array, 'g--', label="CPU")
    plt.legend(loc='best')
    plt.xlabel('InputSize (dim)')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs CUDA GPU RunTime")
    plt.gca().set_xlim((min(dimSize), max(dimSize)))
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_vs_gpuCUDA_plot.png',bbox_inches='tight')
