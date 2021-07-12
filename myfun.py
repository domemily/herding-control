
import random
import numpy as np
import copy
import math

def get_adjacency_matrix(groupA,groupB,sensingRange):
    """
    get adjacency matrix, distance matrix and degree matrix
    groupA is the observed agent swarm
    :return:
    """

    N = groupA.shape[0]
    M = groupB.shape[0]

    #distanceMatrix = np.zeros(N,M)
    #adjacencyMatrix = np.zeros(N,M)
    #degreeMatrix = np.zeros(N,1)

    A = np.tile(groupA, (1, M))
    B = np.tile(groupB.T, (N, 1))
    X = A[:,::2]-B[::2,:]
    Y = A[:,1::2]-B[1::2,:]
    distanceMatrix = (X*X+Y*Y)**0.5
    adjacencyMatrix = copy.deepcopy(distanceMatrix)

    adjacencyMatrix[adjacencyMatrix < sensingRange] = 1
    adjacencyMatrix[adjacencyMatrix >= sensingRange] = 0
    degreeMatrix = np.array([np.sum(adjacencyMatrix, axis=1)]).T

    return distanceMatrix,adjacencyMatrix,degreeMatrix

def get_norm_of_vector(inputvector):
    '''
    get the norm of vector, the size of vector should be N*2
    get the norm of vector in 2D space
    '''
    value = inputvector**2
    norm = (np.array([np.sum(value, axis=1)]).T)**0.5
    return norm

def get_unit_of_vector(inputvector):
    '''
    get the norm of matrix, the size of matrix should be N*2
    get the norm of matrix in 2D space
    '''
    value = inputvector ** 2
    norm = (np.array([np.sum(value, axis=1)]).T) ** 0.5
    norm[norm==0]=10**10
    unit = inputvector/norm
    return unit















