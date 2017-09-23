#!/usr/bin/env python

"""
Vector addition using PyOpenCL.
"""

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'APPLE'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)


# Define the OpenCL kernel you wish to run; most of the interesting stuff you
# will be doing involves modifying or writing kernels:
kernel = """
__kernel void func(__global float* a, __global float* b, __global float* c) {
    unsigned int i = get_global_id(0);
    c[i] = a[i]+b[i];
}
"""

# Load some random data to process. Note that setting the data type is
# important; if your data is stored using one type and your kernel expects a
# different type, your program might either produce the wrong results or fail to
# run.  Note that Numerical Python uses names for certain types that differ from
# those used in OpenCL. For example, np.float32 corresponds to the float type in
# OpenCL:
N = 16
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# We can use PyOpenCL's Array type to easily transfer data from numpy arrays to
# GPU memory (and vice versa):
a_gpu = cl.array.to_device(queue, a)
b_gpu = cl.array.to_device(queue, b)
c_gpu = cl.array.empty(queue, a.shape, a.dtype)

# Launch the kernel; notice that you must specify the global and locals to
# determine how many threads of execution are run. We can take advantage of Numpy to
# use the shape of one of the input arrays as the global size. Since our kernel
# only accesses the global work item ID, we simply set the local size to None:
prg = cl.Program(ctx, kernel).build()
prg.func(queue, a.shape, None, a_gpu.data, b_gpu.data, c_gpu.data)

# Retrieve the results from the GPU:
c = c_gpu.get()

print('input (a):    %s' % a)
print('input (b):    %s' % b)
print('numpy (a+b):  %s' % (a+b))
print('opencl (a+b): %s' % c)

# Compare the results from the GPU with those obtained using Numerical Python;
# this should print True:
print('equal:        %s' % np.allclose(a+b, c))

# Here we compare the speed of performing the vector addition with Python and
# PyOpenCL. Since the execution speed of a snippet of code may vary slightly at
# different times depending on what other things the computer is running, we run
# the operation we wish to time several times and average the results:
M = 3
times = []
for i in range(M):
    start = time.time()
    a+b
    times.append(time.time()-start)
print('python time:  %f' % np.average(times))

times = []
for i in range(M):
    evt = prg.func(queue, a.shape, None, a_gpu.data, b_gpu.data, c_gpu.data)
    evt.wait()
    times.append(1e-9*(evt.profile.end-evt.profile.start))
print('opencl time:  %f' % np.average(times))

# Notice that the execution time of the Python code is actually shorter than
# that of the PyOpenCL code for very short arrays. This is because data
# transfers between host memory and that of the GPU are relatively slow. Try
# gradually increasing the number of elements in a and b up to 100000 and see
# what happens.
