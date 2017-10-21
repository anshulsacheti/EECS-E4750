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

def nonsquare_matrix_mult_opt2(matrix):
    """
    Transpose nonsquare matrix via openCL
    Multiply original by transpose
    Measure runtime of transpose
    Optimization: Using local memory to minimize memory accesses, tiling, increased tile_size
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    #openCL Kernel
    #Naive approach with local/private memory
    #Naive approach reworked to use local memory and tiling
    #Modified boundary condition tiling kernel in lecture
    kernel_code = """
    #define MATRIX_ROW_SIZE {}
    #define MATRIX_COL_SIZE {}
    #define TILE_WIDTH {}
    #define n {}
    __kernel void func(__global float* a, __global float* b) {{

        __local float M[TILE_WIDTH][TILE_WIDTH];
        __local float N[TILE_WIDTH][TILE_WIDTH];

        int bx = get_group_id(0);  int by = get_group_id(1);
        int tx = get_local_id(0); int ty = get_local_id(1);
        int Row = by * get_local_size(1) + ty;
        int Col = bx * get_local_size(0) + tx;
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

            barrier(CLK_LOCAL_MEM_FENCE);

            //Sum tile
            for (int i = 0; i < TILE_WIDTH; ++i) {{
                Cvalue += M[ty][i] * N[i][tx];
            }}

            barrier(CLK_LOCAL_MEM_FENCE);

            //Assign values to output
            if(Row<MATRIX_ROW_SIZE && Col<MATRIX_ROW_SIZE) {{
                b[Row*MATRIX_ROW_SIZE + Col] = Cvalue;

            }}
        }}
    }}
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = cl.array.to_device(queue, matrix_float)
    transposeMult_gpu = cl.array.empty(queue, (matrix.shape[0], matrix.shape[0]), np.float32)
    transposed_gpu  = cl.array.empty(queue, (matrix.shape[1],matrix.shape[0]), np.float32)

    matrix_row_size = matrix.shape[0]
    matrix_col_size = matrix.shape[1]
    TILE_WIDTH = 2

    #Calculate workItems, workGroup size, workGroups for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xWorkItems = int(int(matrix_row_size-1)/TILE_WIDTH)+1
    yWorkItems = int(int(matrix_row_size-1)/TILE_WIDTH)+1
    totalWorkItems = float(TILE_WIDTH*TILE_WIDTH)
    groups = np.int(max(np.ceil(matrix_val_count / xWorkItems),1))

    # print("workItems: %s, matrix_val_count: %s, groups: %s" % (totalWorkItems, matrix_val_count, groups))

    # update template with current runtime requirements
    kernel = kernel_code.format(matrix_row_size, matrix_col_size, TILE_WIDTH, max(matrix_col_size, matrix_row_size))

    #Launch kernel and time it
    #Set global ID, workItems, workGroups
    prg = cl.Program(ctx, kernel).build()
    start = time.time()
    event = prg.func(queue, (xWorkItems*TILE_WIDTH,yWorkItems*TILE_WIDTH,1),(TILE_WIDTH,TILE_WIDTH,1), matrix_gpu.data, transposeMult_gpu.data)
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    # print('openCL_opt2 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('openCL_opt2_transposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    # print('openCL_opt2_mult==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('openCL_opt2 transposed val:\n %s' % transposed)
        print('golden transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('openCL_opt2 mult val:\n %s' % transposedMult)
        print('openCL_opt2 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    # print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult_opt1(matrix):
    """
    Transpose nonsquare matrix via openCL
    Multiply original by transpose
    Measure runtime of transpose
    Optimization: Using local memory to minimize memory accesses
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    #openCL Kernel
    #Naive approach with local/private memory
    kernel_code = """
    #define MATRIX_ROW_SIZE %(matrix_row_size)s
    #define MATRIX_COL_SIZE %(matrix_col_size)s

    __kernel void func(__global float* a, __global float* b, __global float* transposed) {

        unsigned int i = get_local_id(0);
        __local float tmp[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];

        //Initialize tmp to 0
        //Initialize output b to 0 for this thread
        for(int k=0; k<MATRIX_COL_SIZE*MATRIX_ROW_SIZE; k++){
            tmp[k] = 0;
        }

        for(int k=0; k<MATRIX_ROW_SIZE; k++){
            b[k + MATRIX_ROW_SIZE*get_group_id(0)] = 0;
        }

        float localMatrix[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];
        //Copy matrix to local
        for(int j=0; j < MATRIX_COL_SIZE; j++){
            localMatrix[i+get_local_size(0)*get_group_id(0)]=a[i+get_local_size(0)*get_group_id(0)];
        }


        //Transpose output
        transposed[i*MATRIX_ROW_SIZE+get_group_id(0)]=localMatrix[i+get_local_size(0)*get_group_id(0)];

        for(int j=0; j < MATRIX_ROW_SIZE; j++){
                tmp[j+MATRIX_ROW_SIZE*i] = localMatrix[i+get_local_size(0)*get_group_id(0)]*a[i+j*MATRIX_COL_SIZE];
        }

        // Store to output
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
            for(int k=0; k < MATRIX_COL_SIZE; k++){
                if(i==0){
                    b[j + MATRIX_ROW_SIZE*get_group_id(0)] += tmp[j+MATRIX_ROW_SIZE*k];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = cl.array.to_device(queue, matrix_float)
    transposeMult_gpu = cl.array.empty(queue, (matrix.shape[0], matrix.shape[0]), np.float32)
    transposed_gpu  = cl.array.empty(queue, (matrix.shape[1],matrix.shape[0]), np.float32)

    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])

    #Calculate workItems, workGroup size, workGroups for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xWorkItems = min(int(matrix_row_size),1024)
    yWorkItems = min(int(matrix_col_size),1024)
    totalWorkItems = float(xWorkItems*yWorkItems)
    groups = np.int(max(np.ceil(matrix_val_count / xWorkItems),1))

    # print("workItems: %s, matrix_val_count: %s, groups: %s" % (totalWorkItems, matrix_val_count, groups))

    # update template with current runtime requirements
    kernel = kernel_code % {
        'matrix_row_size': matrix_row_size,
        'matrix_col_size': matrix_col_size
        }

    #Launch kernel and time it
    #Set global ID, workItems, workGroups
    prg = cl.Program(ctx, kernel).build()
    start = time.time()
    event = prg.func(queue, (xWorkItems*yWorkItems,1),(groups,1), matrix_gpu.data, transposeMult_gpu.data, transposed_gpu.data)

    #event.wait()
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    # print('openCL_opt1 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('openCL_opt1_transposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    # print('openCL_opt1_mult==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('openCL_opt1 transposed val:\n %s' % transposed)
        print('golden transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('openCL_opt1 mult val:\n %s' % transposedMult)
        print('openCL_opt1 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    # print('--------------------')

    return [transposedMult, runtime]

def nonsquare_matrix_mult(matrix):
    """
    Transpose nonsquare matrix via openCL
    Multiply original by transpose
    Measure runtime of transpose
    Input:
        variable matrix: numpy 2-d array
    Return/Output: [transposed matrix, runtime]
    """

    #Setup openCL
    dev, ctx, queue = setup_CL()

    #openCL Kernel
    #Naive approach
    kernel_code = """
    #define MATRIX_ROW_SIZE %(matrix_row_size)s
    #define MATRIX_COL_SIZE %(matrix_col_size)s

    __kernel void func(__global float* a, __global float* b, __global float* transposed) {

        unsigned int i = get_local_id(0);
        __local float tmp[MATRIX_ROW_SIZE*MATRIX_COL_SIZE];

        //Initialize tmp to 0
        //Initialize output b to 0 for this thread
        for(int k=0; k<MATRIX_COL_SIZE*MATRIX_ROW_SIZE; k++){
            tmp[k] = 0;
        }

        for(int k=0; k<MATRIX_ROW_SIZE; k++){
            b[k + MATRIX_ROW_SIZE*get_group_id(0)] = 0;
        }

        //Transpose output
        transposed[i*MATRIX_ROW_SIZE+get_group_id(0)]=a[i+get_local_size(0)*get_group_id(0)];

        for(int j=0; j < MATRIX_ROW_SIZE; j++){
                tmp[j+MATRIX_ROW_SIZE*i] = a[i+get_local_size(0)*get_group_id(0)]*a[i+j*MATRIX_COL_SIZE];
        }

        // Store to output
        for(int j=0; j < MATRIX_ROW_SIZE; j++){
            for(int k=0; k < MATRIX_COL_SIZE; k++){
                if(i==0){
                    b[j + MATRIX_ROW_SIZE*get_group_id(0)] += tmp[j+MATRIX_ROW_SIZE*k];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    """

    #Move data to device
    matrix_float = matrix.astype(np.float32)
    matrix_gpu = cl.array.to_device(queue, matrix_float)
    transposeMult_gpu = cl.array.empty(queue, (matrix.shape[0], matrix.shape[0]), np.float32)
    transposed_gpu  = cl.array.empty(queue, (matrix.shape[1],matrix.shape[0]), np.float32)

    matrix_row_size = np.int32(matrix.shape[0])
    matrix_col_size = np.int32(matrix.shape[1])

    #Calculate workItems, workGroup size, workGroups for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]
    xWorkItems = min(int(matrix_row_size),1024)
    yWorkItems = min(int(matrix_col_size),1024)
    totalWorkItems = float(xWorkItems*yWorkItems)
    groups = np.int(max(np.ceil(matrix_val_count / xWorkItems),1))

    # print("workItems: %s, matrix_val_count: %s, groups: %s" % (totalWorkItems, matrix_val_count, groups))

    # update template with current runtime requirements
    kernel = kernel_code % {
        'matrix_row_size': matrix_row_size,
        'matrix_col_size': matrix_col_size
        }

    #Launch kernel and time it
    #Set global ID, workItems, workGroups
    prg = cl.Program(ctx, kernel).build()
    start = time.time()
    event = prg.func(queue, (yWorkItems*xWorkItems,1),(groups,1), matrix_gpu.data, transposeMult_gpu.data, transposed_gpu.data)

    #event.wait()
    runtime = time.time()-start

    #Save output
    transposedMult = transposeMult_gpu.get()
    transposed = transposed_gpu.get()

    # print('openCL_opt0 %d x %d transpose-mult time:  %.2E' % (matrix.shape[0], matrix.shape[1], runtime))
    # print('openCL_opt0 transposed==goldenTransposed: %s' % np.allclose(transposed, np.transpose(matrix)))
    # print('openCL_opt0 mult==goldenMult: %s' % np.allclose(transposedMult, matrix.dot(np.transpose(matrix))))
    if not(np.allclose(transposedMult, matrix.dot(np.transpose(matrix)))):
        # print('Original Matrix:\n %s' % matrix)
        print('openCL_opt0 transposed val:\n %s' % transposed)
        print('golden transpose-mult:\n %s' % matrix.dot(np.transpose(matrix)))
        transposedMult[(transposedMult>0) & (transposedMult<1)] = -1
        print('openCL_opt0 mult val:\n %s' % transposedMult)
        print('openCL_opt0 transpose-mult:\n %s' % np.isclose(transposedMult,matrix.dot(np.transpose(matrix))))
    # print('--------------------')

    return [transposedMult, runtime]

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

    #Calculate threads, work size, work groups for input
    matrix_val_count = matrix_float.shape[0]*matrix_float.shape[1]

    xWorkItems = min(int(matrix_row_size),1024)
    yWorkItems = 1
    totalWorkItems = xWorkItems
    groups = np.int(max(np.ceil(matrix_val_count / totalWorkItems),1))

    # print("workItems: %s, matrix_val_count: %s, groups: %s" % (totalWorkItems, matrix_val_count, groups))

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

    # print('-----------------------------')
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
            print("openCL_GPU Square Matrix Transposed Mult time: %.2E" % runtime)
            print("openCL_GPU Square Matrix Transposed Results: %s" % transposed)

        print('openCL_GPUSquareTransposed==goldenSquareTransposed: %s' % np.allclose(transposed, np.transpose(tmp)))

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
    plt.title("pythonCPU Sq Transp RunTime vs openCL GPU Sq Transp RunTime")
    plt.gca().set_xlim((min(squareDimSize), max(squareDimSize)))
    plt.gca().set_ylim([0,0.000005])
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_SquareTranspose_gpuOpenCL_plot.png',bbox_inches='tight')
    plt.close()

    for k in range(1,11):
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

        print('Naive_openCLmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

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

        print('opt1_openCLmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

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

        print('opt2_openCLmult==goldenMult: %s' % np.allclose(transposed, tmp.dot(np.transpose(tmp))))

        dimSize.append(xdim*ydim*k*k)

    print("Avg CPU time: %.2E, Avg naive runtime: %.2E, avg opt1 runtime: %.2E, avg opt2 runtime: %.2E" % (np.average(cpu_runtime_array),np.average(gpu_runtime_array_opt0), np.average(gpu_runtime_array_opt1),np.average(gpu_runtime_array_opt2)))
    #Plot
    plt.gcf()
    # ax = plt.figure().add_subplot(111)
    # ax.plot(dimSize, cpu_runtime_array, 'r--', dimSize, cpu_runtime_array, 'g^')
    plt.plot(dimSize, gpu_runtime_array_opt0, 'r', label="GPU Naive")
    plt.plot(dimSize, gpu_runtime_array_opt1, 'b', label="GPU Opt1")
    plt.plot(dimSize, gpu_runtime_array_opt2, 'o', label="GPU Opt2")
    plt.plot(dimSize, cpu_runtime_array, 'g', label="CPU")
    plt.legend(loc='best')
    plt.xlabel('InputSize')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs openCL GPU RunTime")
    plt.gca().set_xlim((min(dimSize), max(dimSize)))
    plt.gca().set_ylim([0,0.000005])
    plt.autoscale()
    plt.tight_layout()
    plt.ticklabel_format(axis='y',style='sci')
    # ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2e'))
    plt.savefig('pythonCPU_NonSquareTranspose_gpuOpenCL_plot.png',bbox_inches='tight')
