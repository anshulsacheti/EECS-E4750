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
def nonsquare_matrix_mult_opt2(matrix):
    """
    Transpose square matrix via CUDA
    Optimization: Using local memory to minimize memory accesses, tiling
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup CUDA
    #CUDA Kernel
    #Naive approach reworked to use local memory and tiling
    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE %(matrix_row_size)s
    #define MATRIX_COL_SIZE %(matrix_col_size)s
    #define TILE_WIDTH %(TILE_WIDTH)s
    #define n %(n)s
    __global__ void func(float* a, float* b) {

        __shared__ float M[TILE_WIDTH][TILE_WIDTH];
        __shared__ float N[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;  int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;
        int Col = bx * blockDim.x + tx;
        float Cvalue = 0;

        // Loop over the A and B tiles required to compute the C element
        for (int t = 0; t < (n-1)/TILE_WIDTH + 1;++t) {
            if(Row < MATRIX_ROW_SIZE && t*TILE_WIDTH+tx < n) {
                M[ty][tx] = a[Row*n +t*TILE_WIDTH + tx];
            } else {
                M[ty][tx] = 0.0;
            }
            if (t*TILE_WIDTH+ty < n && Col < MATRIX_COL_SIZE) {
                //N[ty][tx] = a[(t*TILE_WIDTH + ty)*MATRIX_COL_SIZE + Col];
                N[ty][tx] = a[t*TILE_WIDTH + MATRIX_ROW_SIZE*Col + ty];
            } else {
                N[ty][tx] = 0.0;
            }
            __syncthreads();
            for (int i = 0; i < TILE_WIDTH; ++i) {
                Cvalue += M[ty][i] * N[i][tx];
            }

            __syncthreads();

            if(Row<MATRIX_ROW_SIZE && Col<MATRIX_COL_SIZE) {
                b[Row*MATRIX_COL_SIZE + Col] = Cvalue;
            }
        }
    }
    """

    kernel_code = """
    #include <stdio.h>
    #define MATRIX_ROW_SIZE {}
    #define MATRIX_COL_SIZE {}
    #define TILE_WIDTH {}
    #define n {}
    __global__ void func(float* a, float* b) {{

        __shared__ float M[TILE_WIDTH][TILE_WIDTH];
        __shared__ float N[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;  int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;
        int Col = bx * blockDim.x + tx;
        float Cvalue = 0;

        // Loop over the A and B tiles required to compute the C element
        for (int t = 0; t < (n-1)/TILE_WIDTH + 1;++t) {{
            if(Row < n && t*TILE_WIDTH+tx < n && tx < MATRIX_COL_SIZE) {{
                M[ty][tx] = a[Row*MATRIX_COL_SIZE +t*TILE_WIDTH + tx];
                printf("M[%d][%d] = %f, Row: %d, tileNum: %d\\n", ty, tx, M[ty][tx], Row, t);
            }} else {{
                M[ty][tx] = 0.0;
                printf("M[%d][%d] = %f, Row: %d, tileProd: %d\\n", ty, tx, M[ty][tx], Row, t*TILE_WIDTH+ty);
            }}
            if (t*TILE_WIDTH+ty < n && Col < n && ty < MATRIX_ROW_SIZE) {{
                //N[ty][tx] = a[(t*TILE_WIDTH + ty)*MATRIX_COL_SIZE + Col];
                N[ty][tx] = a[MATRIX_ROW_SIZE*t/((n-1)/TILE_WIDTH + 1) + t*TILE_WIDTH + MATRIX_COL_SIZE*Col + ty];
                printf("N[%d][%d] = %f, Col: %d, tileNum: %d\\n", ty, tx, N[ty][tx], Col, t);
            }} else {{
                N[ty][tx] = 0.0;
                printf("N[%d][%d] = %f, Col: %d, tileProd: %d\\n", ty, tx, N[ty][tx], Col, t*TILE_WIDTH+ty);
            }}
            __syncthreads();
            for (int i = 0; i < TILE_WIDTH; ++i) {{
                Cvalue += M[ty][i] * N[i][tx];
            }}

            __syncthreads();

            if(Row<n && Col<n) {{
                b[Row*MATRIX_ROW_SIZE + Col] = Cvalue;
                printf("b[%d] = %f, Col: %d, Row: %d, t: %d\\n", (Row*MATRIX_ROW_SIZE + Col), Cvalue, Col, Row, t);
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
    TILE_WIDTH = 2

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    totalThreads = float(TILE_WIDTH*TILE_WIDTH)
    # blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))
    blocks_x = (int(matrix_col_size-1)/TILE_WIDTH)+1
    blocks_y = (int(matrix_row_size-1)/TILE_WIDTH)+1

    print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks_x*blocks_y))

    # update template with current runtime requirements
    kernel = kernel_code.format(matrix_row_size, matrix_col_size, TILE_WIDTH, max(matrix_col_size, matrix_row_size))
    # kernel = kernel_code % {
    #     'matrix_row_size': matrix_row_size,
    #     'matrix_col_size': matrix_col_size,
    #     'TILE_WIDTH': TILE_WIDTH,
    #     'n': max(matrix_col_size, matrix_row_size)
    #     }

    #Compile kernel
    compiled = compiler.SourceModule(kernel)

    #Get compiled kernel
    func = compiled.get_function("func")

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, block = (TILE_WIDTH,TILE_WIDTH,1), grid=(blocks_x,blocks_y,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()

    transformed = np.transpose(matrix)
    print('CUDA_opt2 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    print('CUDAmult_opt2==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(transformed)))
    if not(np.allclose(transposedMult, matrix.dot(transformed))):
        # print('Original Matrix:\n %s' % matrix)
        print('golden opt2 transpose-mult:\n %s' % matrix.dot(transformed))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA opt2 mult val:\n %s' % transposedMult)
        print('CUDA opt2 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(transformed)))
    print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult_opt1(matrix):
    """
    Transpose square matrix via CUDA
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
    __global__ void func(float* a, float* b, float* outputTranspose) {

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
    func = compiled.get_function("func")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = min(int(matrix_col_size),1024)
    totalThreads = float(xThreads*yThreads)
    blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, transposed_gpu, block = (yThreads,1,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    print('CUDA_opt1 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    print('CUDAtransposed_opt1==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    print('CUDAmult_opt1==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('CUDA opt1 transposed val:\n %s' % transposed)
        print('golden opt1 transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA opt1 mult val:\n %s' % transposedMult)
        print('CUDA opt1 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult(matrix):
    """
    Transpose square matrix via CUDA
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
    __global__ void func(float* a, float* b, float* transposed) {

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
    func = compiled.get_function("func")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = min(int(matrix_col_size),1024)
    totalThreads = float(xThreads*yThreads)
    blocks = np.int(max(np.ceil(matrix_val_count / yThreads),1))

    print("threads: %s, matrix_val_count: %s, blocks: %s" % (totalThreads, matrix_val_count, blocks))

    #Launch kernel
    #Number of threads equal to size of name
    start = time.time()
    func(matrix_gpu, transposeMult_gpu, transposed_gpu, block = (yThreads,1,1), grid=(blocks,1,1))
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    print('CUDA %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    print('CUDAtransposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    print('CUDAmult==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('CUDA transposed val:\n %s' % transposed)
        print('golden transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('CUDA mult val:\n %s' % transposedMult)
        print('CUDA transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    print('--------------------')

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
    kernel = """
    __global__ void func(float* a, float* b, const int matrix_row_size) {
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
    func = compiled.get_function("func")

    #Calculate threads, block size, blocks for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xThreads = min(int(matrix_row_size),1024)
    yThreads = 1
    totalThreads = float(xThreads)#*yThreads)
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
        # print('Original Matrix:\n %s' % matrix)
        print('golden transpose:\n %s' % np.transpose(matrix))
        transposed[(transposed>0) & (transposed<1)] = -1
        print('CUDA val:\n %s' % transposed)
        print('CUDA transpose: %s' % np.isclose(np.transpose(matrix),transposed))
    print('--------------------')

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

    gpu_transpose_array_opt0 = []
    gpu_runtime_array_opt0 = []

    gpu_transpose_array_opt1 = []
    gpu_runtime_array_opt1 = []

    gpu_transpose_array_opt2 = []
    gpu_runtime_array_opt2 = []

    dimSize = []

    #Handle square, nonsquare matrices
    if args.dim2!=0:

        for k in range(1,3):
            tmp = np.zeros([args.dim1*k, args.dim2*k])
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i,j] = i*tmp.shape[1]+j

            print('INPUT:\n %s' % tmp)
            # # CPU Runtime
            # transposed, runtime = python_nonsquare_matrix_mult(tmp)
            # cpu_transpose_array.append(transposed)
            # cpu_runtime_array.append(runtime)
            #
            # #GPU Runtime
            # transposed0, runtime = nonsquare_matrix_mult(tmp)
            # gpu_transpose_array_opt0.append(transposed)
            # gpu_runtime_array_opt0.append(runtime)
            #
            # #GPU Opt1 Runtime
            # transposed, runtime = nonsquare_matrix_mult_opt1(tmp)
            # gpu_transpose_array_opt1.append(transposed)
            # gpu_runtime_array_opt1.append(runtime)

            #GPU Opt2 Runtime
            transposed, runtime = nonsquare_matrix_mult_opt2(tmp)
            gpu_transpose_array_opt2.append(transposed)
            gpu_runtime_array_opt2.append(runtime)

            dimSize.append(args.dim1*args.dim2*k*k)

    else:
        for k in range(1,11):

            #Generate input
            # tmp = np.random.rand(args.dim1*i, args.dim1*i)
            tmp = np.zeros([args.dim1*k, args.dim1*k])
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    tmp[i,j] = i*tmp.shape[0]+j

            #CPU Runtime
            transposed, runtime = python_square_matrix(tmp)
            cpu_transpose_array.append(transposed)
            cpu_runtime_array.append(runtime)

            #GPU Runtime
            transposed, runtime = transpose_square_matrix(tmp)
            gpu_transpose_array_opt0.append(transposed)
            gpu_runtime_array_opt0.append(runtime)

            dimSize.append(args.dim1*args.dim2*k*k)

    # print("Avg naive runtime: %.2E, avg opt1 runtime: %.2E, avg opt2 runtime: %.2E" % (np.average(gpu_runtime_array_opt0), np.average(gpu_runtime_array_opt1),np.average(gpu_runtime_array_opt2)))
    # #Plot
    # plt.gcf()
    # # ax = plt.figure().add_subplot(111)
    # # ax.plot(dimSize, gpu_runtime_array, 'r--', dimSize, cpu_runtime_array, 'g^')
    # plt.plot(dimSize, gpu_runtime_array_opt0, 'r', label="GPU Naive")
    # plt.plot(dimSize, gpu_runtime_array_opt1, 'b', label="GPU Opt1")
    # # plt.plot(dimSize, gpu_opt2_runtime_array, 'o', label="GPU Naive")
    # # plt.plot(dimSize, cpu_runtime_array, 'g', label="CPU")
    # plt.legend(loc='best')
    # plt.xlabel('InputSize (dim)')
    # plt.ylabel('RunTime (s)')
    # plt.title("pythonCPU RunTime vs CUDA GPU RunTime")
    # plt.gca().set_xlim((min(dimSize), max(dimSize)))
    # plt.gca().set_ylim([0,0.000005])
    # plt.autoscale()
    # plt.tight_layout()
    # plt.ticklabel_format(axis='y',style='sci')
    # # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    # plt.savefig('pythonCPU_vs_gpuCUDA_plot.png',bbox_inches='tight')
