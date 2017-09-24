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

    plt.gcf()
    ax = plt.figure().add_subplot(111)
    ax.plot(nameLength, timeArray)
    plt.xlabel('InputSize (number of chars)')
    plt.ylabel('RunTime (s)')
    plt.title("pythonCPU RunTime vs InputSize")
    plt.gca().set_xlim((min(nameLength), max(nameLength)))
    plt.autoscale()
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3e'))
    plt.savefig('CPU_plot.png',bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name')
    parser.add_argument('--multiIter', type=int)

    args = parser.parse_args()
    if args.multiIter:
        python_multi_hash(list(args.name),args.multiIter)
    else:
        python_simple_hash(list(args.name))
