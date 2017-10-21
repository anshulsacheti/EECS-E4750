#!/usr/bin/env python
import time
import argparse

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

import numpy as np
np.set_printoptions(suppress=True)
import string
import random
import os

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pdb
def nonsquare_matrix_mult_opt3(matrix):
    """
    Transpose nonsquare matrix via CUDA
    Multiply original by transpose
    Measure runtime of transpose
    Optimization: Using local memory to minimize memory accesses, tiling, increased tile_size
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    #Naive approach reworked to use local memory and tiling
    #Modified boundary condition tiling kernel in lecture
    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE {}
    #define MATRIX_COL_SIZE {}
    #define TILE_WIDTH {}
    #define n {}
    __global__ void opt3(float* a, float* b) {{

        __shared__ float M[TILE_WIDTH][TILE_WIDTH];
        __shared__ float N[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;  int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;
        int Col = bx * blockDim.x + tx;
        float Cvalue = 0;

        // Loop over the A and B tiles required to compute the C element
        for (int t = 0; t < (n-1)/TILE_WIDTH + 1;++t) {{

            //Assign rows of input
            if(t*TILE_WIDTH+tx < MATRIX_COL_SIZE && tx < MATRIX_COL_SIZE && (Row*MATRIX_COL_SIZE + t*TILE_WIDTH + tx)<MATRIX_COL_SIZE*MATRIX_ROW_SIZE) {{
                    M[ty][tx] = a[Row*MATRIX_COL_SIZE + t*TILE_WIDTH + tx];
            }} else {{
                M[ty][tx] = 0.0;
            }}

            //Assign columns of transpose
            if (t*TILE_WIDTH+ty < n && Col < MATRIX_ROW_SIZE) {{
                N[ty][tx] = a[t*TILE_WIDTH + MATRIX_COL_SIZE*Col + ty];
            }} else {{
                N[ty][tx] = 0.0;
            }}

            __syncthreads();

            //Sum tile
            for (int i = 0; i < TILE_WIDTH; ++i) {{
                Cvalue += M[ty][i] * N[i][tx];
            }}

            __syncthreads();

            //Assign values to output
            if(Row<MATRIX_ROW_SIZE && Col<MATRIX_ROW_SIZE) {{
                b[Row*MATRIX_ROW_SIZE + Col] = Cvalue;

            }}
        }}
    }}
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix_float)
    transposeMult_gpu = gpuarray.empty((matrix.shape[0],matrix.shape[0]), np.float32)
    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])
    TILE_WIDTH = 32

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    totalThreads = float(TILE_WIDTH*TILE_WIDTH)
    # blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    # Multiplying matrix MxN by NxM yielding MxM -> need number of threads >= num elements in matrix
    blocks_x = (int(matrix_row_size-1)/TILE_WIDTH)+1
    blocks_y = (int(matrix_row_size-1)/TILE_WIDTH)+1

    # print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks_x*blocks_y))

    # update template with current runtime requirements
    kernel = kernel_code.format(matrix_row_size, matrix_col_size, TILE_WIDTH, max(matrix_col_size, matrix_row_size))

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("opt3")

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, block = (TILE_WIDTH,TILE_WIDTH,1), grid=(blocks_x,blocks_y,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()

    transformed = np.transpose(matrix)
    # print('CUDA_opt2 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('CUDAmult_opt2==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(transformed)))
    # print('golden opt2 transpose-mult:\n %s' % matrix.dot(transformed))
    # print('CUDA opt2 mult val:\n %s' % transposedMult)
    if not(np.allclose(transposedMult, matrix.dot(transformed))):
        # print('Original Matrix:\n %s' % matrix)
        print('golden opt2 transpose-mult:\n %s' % matrix.dot(transformed))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA opt2 mult val:\n %s' % transposedMult)
        print('CUDA opt2 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(transformed)))
    # print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult_opt2(matrix):
    """
    Transpose nonsquare matrix via CUDA
    Multiply original by transpose
    Measure runtime of transpose
    Optimization: Using local memory to minimize memory accesses, tiling
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    #Naive approach reworked to use local memory and tiling
    #Modified boundary condition tiling kernel in lecture
    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE {}
    #define MATRIX_COL_SIZE {}
    #define TILE_WIDTH {}
    #define n {}
    __global__ void opt2(float* a, float* b) {{

        __shared__ float M[TILE_WIDTH][TILE_WIDTH];
        __shared__ float N[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;  int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;
        int Col = bx * blockDim.x + tx;
        float Cvalue = 0;

        // Loop over the A and B tiles required to compute the C element
        for (int t = 0; t < (n-1)/TILE_WIDTH + 1;++t) {{

            //Assign rows of input
            if(t*TILE_WIDTH+tx < MATRIX_COL_SIZE && tx < MATRIX_COL_SIZE && (Row*MATRIX_COL_SIZE + t*TILE_WIDTH + tx)<MATRIX_COL_SIZE*MATRIX_ROW_SIZE) {{
                    M[ty][tx] = a[Row*MATRIX_COL_SIZE + t*TILE_WIDTH + tx];
            }} else {{
                M[ty][tx] = 0.0;
            }}

            //Assign columns of transpose
            if (t*TILE_WIDTH+ty < n && Col < MATRIX_ROW_SIZE) {{
                N[ty][tx] = a[t*TILE_WIDTH + MATRIX_COL_SIZE*Col + ty];
            }} else {{
                N[ty][tx] = 0.0;
            }}

            __syncthreads();

            //Sum tile
            for (int i = 0; i < TILE_WIDTH; ++i) {{
                Cvalue += M[ty][i] * N[i][tx];
            }}

            __syncthreads();

            //Assign values to output
            if(Row<MATRIX_ROW_SIZE && Col<MATRIX_ROW_SIZE) {{
                b[Row*MATRIX_ROW_SIZE + Col] = Cvalue;

            }}
        }}
    }}
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix_float)
    transposeMult_gpu = gpuarray.empty((matrix.shape[0],matrix.shape[0]), np.float32)
    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])
    TILE_WIDTH = 8

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    totalThreads = float(TILE_WIDTH*TILE_WIDTH)
    # blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    # Multiplying matrix MxN by NxM yielding MxM -> need number of threads >= num elements in matrix
    blocks_x = (int(matrix_row_size-1)/TILE_WIDTH)+1
    blocks_y = (int(matrix_row_size-1)/TILE_WIDTH)+1

    # print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks_x*blocks_y))

    # update template with current runtime requirements
    kernel = kernel_code.format(matrix_row_size, matrix_col_size, TILE_WIDTH, max(matrix_col_size, matrix_row_size))

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("opt2")

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, block = (TILE_WIDTH,TILE_WIDTH,1), grid=(blocks_x,blocks_y,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()

    transformed = np.transpose(matrix)
    # print('CUDA_opt2 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('CUDAmult_opt2==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(transformed)))
    # print('golden opt2 transpose-mult:\n %s' % matrix.dot(transformed))
    # print('CUDA opt2 mult val:\n %s' % transposedMult)
    if not(np.allclose(transposedMult, matrix.dot(transformed))):
        # print('Original Matrix:\n %s' % matrix)
        print('golden opt2 transpose-mult:\n %s' % matrix.dot(transformed))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA opt2 mult val:\n %s' % transposedMult)
        print('CUDA opt2 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(transformed)))
    # print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult_opt1(matrix):
    """
    Transpose nonsquare matrix via CUDA
    Multiply original by transpose
    Measure runtime of transpose
    Optimization: Using local memory to minimize memory accesses
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    #Naive approach + local memory storage
    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE %(matrix_row_size)s
    #define MATRIX_COL_SIZE %(matrix_col_size)s
    __global__ void opt1(float* a, float* b, float* outputTranspose) {

        unsigned int i = threadIdx.x;
        __shared__ float tmp[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];

        //Initialize tmp to 0
        //Initialize output b to 0 for this thread
        for(int k=0; k<MATRIX_COL_SIZE*MATRIX_ROW_SIZE; k++){
            tmp[k] = 0;
        }

        for(int k=0; k<MATRIX_ROW_SIZE; k++){
            b[k + MATRIX_ROW_SIZE*blockIdx.x] = 0;
        }

        float localMatrix[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];
        //Copy matrix to local
        for(int j=0; j < MATRIX_COL_SIZE; j++){
            localMatrix[i+blockDim.x*blockIdx.x]=a[i+blockDim.x*blockIdx.x];
        }

        //__syncthreads();

        //Transpose output
        outputTranspose[i*MATRIX_ROW_SIZE+blockIdx.x]=localMatrix[i+blockDim.x*blockIdx.x];

        //Use shared memory to do multiply
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
                tmp[j+MATRIX_ROW_SIZE*i] = localMatrix[i+blockDim.x*blockIdx.x]*a[i+j*MATRIX_COL_SIZE];
                //outputTranspose[j+i*MATRIX_ROW_SIZE];//*
        }

        //__syncthreads();
        // Store to output
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
            for(int k=0; k < MATRIX_COL_SIZE; k++){
                if(i==0){
                    b[j + MATRIX_ROW_SIZE*blockIdx.x] += tmp[j+MATRIX_ROW_SIZE*k];
                }
            }
        }
        __syncthreads();
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix_float)
    transposeMult_gpu = gpuarray.empty((matrix.shape[0],matrix.shape[0]), np.float32)
    transposed_gpu  = gpuarray.empty((matrix.shape[1],matrix.shape[0]), np.float32)
    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])

    # update template with current runtime requirements
    kernel = kernel_code % {
        'matrix_row_size': matrix_row_size,
        'matrix_col_size': matrix_col_size
        }

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("opt1")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = min(int(matrix_col_size),1024)
    totalThreads = float(xThreads*yThreads)
    blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    # print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, transposed_gpu, block = (yThreads,1,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    # print('CUDA_opt1 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('CUDAtransposed_opt1==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    # print('CUDAmult_opt1==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('CUDA opt1 transposed val:\n %s' % transposed)
        print('golden opt1 transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA opt1 mult val:\n %s' % transposedMult)
        print('CUDA opt1 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    # print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult(matrix):
    """
    Transpose nonsquare matrix via CUDA
    Multiply original by transpose
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    #Naive approach
    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE %(matrix_row_size)s
    #define MATRIX_COL_SIZE %(matrix_col_size)s
    __global__ void opt0(float* a, float* b, float* transposed) {

        unsigned int i = threadIdx.x;
        __shared__ float tmp[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];

        //Initialize tmp to 0
        //Initialize output b to 0 for this thread
        for(int k=0; k<MATRIX_COL_SIZE*MATRIX_ROW_SIZE; k++){
            tmp[k] = 0;
        }

        for(int k=0; k<MATRIX_ROW_SIZE; k++){
            b[k + MATRIX_ROW_SIZE*blockIdx.x] = 0;
        }

        //__syncthreads();

        //Transpose output
        transposed[i*MATRIX_ROW_SIZE+blockIdx.x]=a[i+blockDim.x*blockIdx.x];

        //Calculate transpose
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
                tmp[j+MATRIX_ROW_SIZE*i] = a[i+blockDim.x*blockIdx.x]*a[i+j*MATRIX_COL_SIZE];
        }

        //__syncthreads();
        // Store to output
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
            for(int k=0; k < MATRIX_COL_SIZE; k++){
                if(i==0){
                    b[j + MATRIX_ROW_SIZE*blockIdx.x] += tmp[j+MATRIX_ROW_SIZE*k];
                }
            }
        }
        __syncthreads();
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = gpuarray.to_gpu(matrix_float)
    transposeMult_gpu = gpuarray.empty((matrix.shape[0],matrix.shape[0]), np.float32)
    transposed_gpu  = gpuarray.empty((matrix.shape[1],matrix.shape[0]), np.float32)
    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])

    # update template with current runtime requirements
    kernel = kernel_code % {
        'matrix_row_size': matrix_row_size,
        'matrix_col_size': matrix_col_size
        }

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("opt0")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = min(int(matrix_col_size),1024)
    totalThreads = float(xThreads*yThreads)
    blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    # print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, transposed_gpu, block = (yThreads,1,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    # print('CUDA %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('CUDAtransposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    # print('CUDAmult==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('CUDA transposed val:\n %s' % transposed)
        print('golden transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA mult val:\n %s' % transposedMult)
        print('CUDA transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    # print('--------------------')

    return [transposedMult, runtime]

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
    # Simple transpose
    kernel = """
    __global__ void square(float* a, float* b, const int matrix_row_size) {
        unsigned int i = threadIdx.x;
        b[i*blockDim.x+blockIdx.x]=a[i+blockDim.x*blockIdx.x];
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
    func = compiled.get_function("square")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = 1
    totalThreads = float(xThreads)#*yThreads)
    blocks = np.int(max(np.ceil(matrix_val_count / totalThreads),1))

    # print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transpose_gpu, matrix_row_size, block = (xThreads,yThreads,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposed = transpose_gpu.get()

    # print('CUDA %d x %d transpose time:  %.2E' % (matrix.shape[0], matrix.shape[0], runtime))
    # print('CUDA==golden: %s' % np.allclose(transposed, np.transpose(matrix)))
    if not(np.allclose(transposed, np.transpose(matrix))):
        # print('Original Matrix:\n %s' % matrix)
        print('golden transpose:\n %s' % np.transpose(matrix))
        transposed[(transposed>0) & (transposed<1)] = -1
        print('CUDA val:\n %s' % transposed)
        print('CUDA transpose: %s' % np.isclose(np.transpose(matrix),transposed))
    # print('--------------------')

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

    # print('python transpose time:  %.2E' % end)
    return [transposed_matrix, end]

def python_nonsquare_matrix_mult(matrix):
    """
    Calculate transpose of nonsquare matrix MxN
    Multiply transpose by non-transpose
    Measure runtime of overall calculation
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    transposed_matrix = np.zeros([matrix.shape[1],matrix.shape[0]])
    start = time.time()
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         transposed_matrix[j,i] = matrix[i,j]

    transposed_matrix = np.transpose(matrix)
    product = matrix.dot(transposed_matrix)

    # transposed_matrix = np.transpose(matrix)
    end = time.time()-start

    # print("Python Golden Transpose: %s" % product)
    # print('python transpose time:  %.2E' % end)
    return [product, end]

if __name__=="__main__":

    # #Handle command line inputs
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument("dim1", type=int)
    # parser.add_argument("dim2", type=int, default=0)
    #
    # args = parser.parse_args()

    # Starting dims
    xdim=10
    ydim=11

    #initialize arrays
    cpu_squareTranspose_array = []
    cpu_squareRuntime_array = []

    gpu_squareTranspose_array = []
    gpu_squareRuntime_array = []

    cpu_transpose_array = []
    cpu_runtime_array = []

    gpu_transpose_array_opt0 = []
    gpu_runtime_array_opt0 = []

    gpu_transpose_array_opt1 = []
    gpu_runtime_array_opt1 = []

    gpu_transpose_array_opt2 = []
    gpu_runtime_array_opt2 = []

    gpu_transpose_array_opt3 = []
    gpu_runtime_array_opt3 = []

    dimSize = []
    squareDimSize = []

    #Handle square, nonsquare matrices
    for k in range(1,11):

        #Generate input
        # tmp = np.random.rand(xdim*i, xdim*i)
        tmp = np.zeros([xdim*k, xdim*k])
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i,j] = i*tmp.shape[0]+j

        print("Input DIM: [%d,%d]" % (tmp.shape[0], tmp.shape[1]))

        #CPU Runtime
        transposed, runtime = python_square_matrix(tmp)
        cpu_squareRuntime_array.append(runtime)

        if k<3:
            print("CPU Square Matrix Transposed Mult time: %.2E" % runtime)
            print("CPU Square Matrix Transposed Results: %s" % transposed)

        print('CPUSquareTransposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(tmp)))

        #GPU Runtime
        transposed, runtime = transpose_square_matrix(tmp)
        gpu_squareRuntime_array.append(runtime)

        if k<3:
            print("CUDA_GPU Square Matrix Transposed Mult time: %.2E" % runtime)
            print("CUDA_GPU Square Matrix Transposed Results: %s" % transposed)

        print('CUDA_GPUSquareTransposed==goldenSquareTransposed: %s' % np.allclose(transposed, np.transpose(tmp)))

        squareDimSize.append(xdim*k*xdim*k)

    print("Avg SquareTranspose CPU time: %.2E, Avg SquareTranspose GPU runtime: %.2E" % (np.average(cpu_squareRuntime_array),np.average(gpu_squareRuntime_array)))

    #Plot
    plt.gcf()
    # ax = plt.figure().add_subplot(111)
    # ax.plot(dimSize, cpu_runtime_array, 'r--', dimSize, cpu_runtime_array, 'g^')
    plt.plot(squareDimSize, gpu_squareRuntime_array, 'r', label="GPU")
    plt.plot(squareDimSize, cpu_squareRuntime_array, 'g', label="CPU")
    plt.legend(loc='best')
    plt.xlabel('InputSize')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU Sq Transp RunTime vs CUDA GPU Sq Transp RunTime")
    plt.gca().set_xlim((min(squareDimSize), max(squareDimSize)))
    plt.gca().set_ylim([0,0.000005])
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_SquareTranspose_gpuCUDA_plot.png',bbox_inches='tight')
    plt.close()

    for k in range(1,11):
        # dims = [[0], [2,3], [3,2], [3,4], [4,3]]
        # tmp = np.zeros([dims[k][0],dims[k][1]])
        # for i in range(tmp.shape[0]):
        #     for j in range(tmp.shape[1]):
        #         tmp[i,j] = i*tmp.shape[1]+j
        tmp = np.zeros([xdim*k, ydim*k])
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i,j] = i*tmp.shape[1]+j

        # print('INPUT:\n %s' % tmp)
        print("Input DIM: [%d,%d]" % (tmp.shape[0], tmp.shape[1]))
        # CPU Runtime
        loops = 10

        times = []
        transposed = []
        for i in range(loops):
            transposed, runtime = python_nonsquare_matrix_mult(tmp)
            times.append(runtime)

        runtime = np.average(times)
        cpu_transpose_array.append(transposed)
        cpu_runtime_array.append(runtime)

        if k<3:
            print("CPU Matrix Mult time: %.2E" % runtime)
            print("CPU Matrix Results: %s" % transposed)

        print('CPUmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        #GPU Runtime
        times = []
        transposed = []
        for i in range(loops):
            transposed, runtime = nonsquare_matrix_mult(tmp)
            times.append(runtime)

        runtime = np.average(times)
        gpu_transpose_array_opt0.append(transposed)
        gpu_runtime_array_opt0.append(runtime)

        if k<3:
            print("Naive Matrix Mult time: %.2E" % runtime)
            print("Naive Matrix Results: %s" % transposed)

        print('Naive_CUDAmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        #GPU Opt1 Runtime
        times = []
        transposed = []
        for i in range(loops):
            transposed, runtime = nonsquare_matrix_mult_opt1(tmp)
            times.append(runtime)

        runtime = np.average(times)
        gpu_transpose_array_opt1.append(transposed)
        gpu_runtime_array_opt1.append(runtime)

        if k<3:
            print("opt1_Matrix Mult time: %.2E" % runtime)
            print("opt1_Matrix Results: %s" % transposed)

        print('opt1_CUDAmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        #GPU Opt2 Runtime
        times = []
        transposed = []
        for i in range(loops):
            transposed, runtime = nonsquare_matrix_mult_opt2(tmp)
            times.append(runtime)

        runtime = np.average(times)
        gpu_transpose_array_opt2.append(transposed)
        gpu_runtime_array_opt2.append(runtime)

        if k<3:
            print("opt2_Matrix Mult time: %.2E" % runtime)
            print("opt2_Matrix Results: %s" % transposed)

        print('opt2_CUDAmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        #GPU Opt3 Runtime
        times = []
        transposed = []
        for i in range(loops):
            transposed, runtime = nonsquare_matrix_mult_opt3(tmp)
            times.append(runtime)

        runtime = np.average(times)
        gpu_transpose_array_opt3.append(transposed)
        gpu_runtime_array_opt3.append(runtime)

        if k<3:
            print("opt3_Matrix Mult time: %.2E" % runtime)
            print("opt3_Matrix Results: %s" % transposed)

        print('opt3_CUDAmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        dimSize.append(xdim*ydim*k*k)

    print("Avg CPU time: %.2E, Avg naive runtime: %.2E, avg opt1 runtime: %.2E, avg opt2 runtime: %.2E, avg opt3 runtime: %.2E" % (np.average(cpu_runtime_array),np.average(gpu_runtime_array_opt0), np.average(gpu_runtime_array_opt1),np.average(gpu_runtime_array_opt2),np.average(gpu_runtime_array_opt3)))
    #Plot
    plt.gcf()
    # ax = plt.figure().add_subplot(111)
    # ax.plot(dimSize, cpu_runtime_array, 'r--', dimSize, cpu_runtime_array, 'g^')
    plt.plot(dimSize, gpu_runtime_array_opt0, 'r', label="GPU Naive")
    plt.plot(dimSize, gpu_runtime_array_opt1, 'b', label="GPU Opt1")
    plt.plot(dimSize, gpu_runtime_array_opt2, 'o', label="GPU Opt2")
    plt.plot(dimSize, gpu_runtime_array_opt3, 'p', label="GPU Opt3")
    plt.plot(dimSize, cpu_runtime_array, 'g', label="CPU")
    plt.legend(loc='best')
    plt.xlabel('InputSize')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs CUDA GPU RunTime")
    plt.gca().set_xlim((min(dimSize), max(dimSize)))
    plt.gca().set_ylim([0,0.000005])
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_NonSquareTranspose_gpuCUDA_plot.png',bbox_inches='tight')
