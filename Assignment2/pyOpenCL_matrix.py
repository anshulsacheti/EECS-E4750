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
        unsigned int i = get_global_id(0);
        unsigned int j = get_global_id(1);
        b[i*matrix_row_size+j]=a[i+j*matrix_row_size];
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = cl.array.to_device(queue, matrix_float)
    transpose_gpu = cl.array.empty(queue, matrix.shape, matrix_float.dtype)

    matrix_row_size = np.int32(matrix.shape[1])
    # mf = cl.mem_flags
    # matrix_row_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix_row_size)

    # matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    # blockThreads = 1024.0
    # blocks = np.int(max(np.ceil(matrix_val_count / blockThreads),1))
    # print("matrix_val_count: %s, blocks: %s" % (matrix_val_count, blocks))

    #Launch kernel and time it
    #Set global ID
    prg = cl.Program(ctx, kernel).build()
    start = time.time()
    event = prg.func(queue, matrix_float.shape, None, matrix_gpu.data, transpose_gpu.data, matrix_row_size)
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
        print('golden transpose: %s' % np.transpose(matrix))
        print('openCL transpose: %s' % transposed)

    print('-----------------------------')
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
        for i in range(1,11):
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
        plt.title("pythonCPU RunTime vs openCL GPU RunTime")
        plt.gca().set_xlim((min(dimSize), max(dimSize)))
        plt.autoscale()
        plt.tight_layout()
        plt.ticklabel_format(axis='y',style='sci')
        # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
        plt.savefig('pythonCPU_vs_gpuOpenCL_plot.png',bbox_inches='tight')
