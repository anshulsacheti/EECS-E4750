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
        #Easy switching for local vs remote machine
        if p.name == 'Apple':
            NAME = 'Apple'
        if p.name == NAME:
            dev = p.get_devices()

    # Command queue, enable GPU profiling
    ctx = cl.Context(dev)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

    return [dev,ctx,queue]

def transpose_square_matrix(matrix):
    """
    Transpose square matrix via openCL
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    #openCL Kernel
    kernel = """
    __kernel void func(__global float* a, __global float* b, const int matrix_row_size) {
        int i = get_local_id(0);
        b[i*get_local_size(0)+get_group_id(0)]=a[i+get_local_size(0)*get_group_id(0)];
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = cl.array.to_device(queue, matrix_float)
    transpose_gpu = cl.array.empty(queue, matrix.shape, matrix_float.dtype)

    matrix_row_size = np.int32(matrix.shape[1])
    #
    #Calculate threads, work size, work groups for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    # threadCounts = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # #blockThreads = 8
    xWorkItems = min(int(matrix_row_size),1024)
    yWorkItems = 1
    totalWorkItems = xWorkItems
    groups = np.int(max(np.ceil(matrix_val_count / totalWorkItems),1))
    #
    print("workItems: %s, matrix_val_count: %s, groups: %s" % (totalWorkItems, matrix_val_count, groups))

    #Launch kernel
    #Number of threads equal to size of name
    # start = time.time()
    # func(matrix_gpu, transpose_gpu, matrix_row_size, block = (xThreads,yThreads,1), grid=(blocks,1,1))

    #Launch kernel and time it
    #Set global ID
    prg = cl.Program(ctx, kernel).build()
    start = time.time()
    event = prg.func(queue, (xWorkItems*xWorkItems,1),(groups,1), matrix_gpu.data, transpose_gpu.data, matrix_row_size)
    #event.wait()
    runtime = time.time()-start
    #Save output
    transposed = transpose_gpu.get()

    # print('Original Matrix: %s' % matrix)
    # print('golden transpose: %s' % np.transpose(matrix))
    # print('openCL transpose: %s' % transposed)
    print('opencl %d x %d transpose time:  %.2E' % (matrix.shape[0], matrix.shape[0], runtime))
    print('openCL==golden: %s' % np.allclose(transposed, np.transpose(matrix)))
    if not(np.allclose(transposed, np.transpose(matrix))):
        # print('Original Matrix:\n %s' % matrix)
        print('golden transpose:\n %s' % np.transpose(matrix))
        transposed[(transposed>0) & (transposed<1)] = -1
        print('openCL val:\n %s' % transposed)
        print('openCL transpose: %s' % np.isclose(np.transpose(matrix),transposed))

    print('-----------------------------')
    return [transposed, runtime]

def python_square_matrix(matrix):
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
        for i in range(1,11):

            #Generate input
            # tmp = np.random.rand(args.dim1*i, args.dim1*i)
            tmp = np.zeros([args.dim1*i, args.dim1*i])
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i,j] = i*tmp.shape[0]+j

            #CPU Runtime
            transposed, runtime = python_square_matrix(tmp)
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
        plt.title("pythonCPU RunTime vs openCL GPU RunTime")
        plt.gca().set_xlim((min(dimSize), max(dimSize)))
        plt.autoscale()
        plt.tight_layout()
        plt.ticklabel_format(axis='y',style='sci')
        # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
        plt.savefig('pythonCPU_vs_gpuOpenCL_plot.png',bbox_inches='tight')
