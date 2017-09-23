import argparse
import pdb
import numpy as np

def simple_hash(name):
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
    print(hashed)

def multi_hash(name,iterCount):
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
    refName = name
    for i in range(iterCount):
        hashed = np.zeros(len(name)*iterCount).astype(int)
        count=0
        name = refName*(i+1)
        for char in name:
            hashed[count]=ord(char) % 17
            count+=1

    print(hashed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name')
    parser.add_argument('--multiIter', type=int)

    args = parser.parse_args()
    if args.multiIter:
        multi_hash(list(args.name),args.multiIter)
    else:
        simple_hash(list(args.name))
