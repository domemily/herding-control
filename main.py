# written by Shuai Zhang 2021-7-7
# this is a simple pursuit-evasion game or herding control game by using matrix
# swarm behavior of prey: refer to
# Chen, Y., & Kolokolnikov, T. (2014). A minimal model of predator–swarm interactions.
import random
import numpy as np
import myfun as mf
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import time
import pygame, sys
from pygame.locals import *

# tic toc process
p0 = time.process_time()

# initialization

N = 100
M=5
maxstep= 3000

prey_position= np.zeros((N,2,maxstep+1))
predator_position = np.zeros((M,2,maxstep+1))


sensingRange_prey = 5
sensingRange_predator = 5
pushRange=0.1

parameter_alignment_strength = 1
parameter_flight_strength = 1
parameter_pursuit_strength = 2.5
parameter_repulsion_strength = 1
parameter_push_strength = 2.5

maxVelocity_prey = 1
maxVelocity_predator = 1.2
deltaT = 0.05

# initial position of predator and prey

prey_position[:,:,0] = 1*np.random.randn(N,2)+20
predator_position[:,:,0] = 1*np.random.randn(M,2)+20
target_position = [0,0]
print(prey_position)
print(predator_position)

# the main loop
current_step=0
while current_step < maxstep:

    #prey maneuverability
    groupA = copy.deepcopy(prey_position[:,:,current_step])
    groupB = copy.deepcopy(predator_position[:,:,current_step])

    #flight force of prey to predators
    [prey_to_predator_distanceMatrix, prey_to_predator_adjacencyMatrix, prey_to_predator_degreeMatrix] = mf.get_adjacency_matrix(groupA,groupB,sensingRange_prey)
    A = np.tile(groupA, (1, M))
    BB=groupB.reshape(1,-1)
    B = np.tile(np.array(BB),(N,1))

    H = np.zeros((N,2*M))
    H[:,::2] = prey_to_predator_adjacencyMatrix
    H[:,1::2] = prey_to_predator_adjacencyMatrix

    Z = np.zeros((N,2*M))
    Z[:,::2]=prey_to_predator_distanceMatrix**3
    Z[:,1::2]=prey_to_predator_distanceMatrix**3

    Z[Z==0]=10**100 # Z cannot be zero


    C = (H*(A-B))/Z
    C=np.nan_to_num(C)

    C_x = C[:,::2]
    C_y=C[:,1::2]
    D_x=np.array([np.sum(C_x, axis=1)]).T
    D_y=np.array([np.sum(C_y, axis=1)]).T
    D=np.concatenate((D_x, D_y), axis = 1)
    prey_flight = parameter_flight_strength/prey_to_predator_degreeMatrix*D
    prey_flight=np.nan_to_num(prey_flight)

    # alignment force of prey to predators
    [prey_to_prey_distanceMatrix, prey_to_prey_adjacencyMatrix, prey_to_prey_degreeMatrix] = mf.get_adjacency_matrix(groupA, groupA,sensingRange_prey)

    A = np.tile(groupA, (1, N))
    BB=groupA.reshape(1,-1)
    B = np.tile(np.array(BB),(N,1))

    H = np.zeros((N, 2 * N))
    H[:, ::2] = prey_to_prey_adjacencyMatrix
    H[:, 1::2] = prey_to_prey_adjacencyMatrix

    Z = np.zeros((N, 2 * N))
    Z[:, ::2] = prey_to_prey_distanceMatrix ** 2
    Z[:, 1::2] = prey_to_prey_distanceMatrix ** 2

    Z[Z==0]=10**100  #


    C = H*((A-B)/Z- parameter_alignment_strength*(A-B))
    C=np.nan_to_num(C)

    C_x = C[:, ::2]
    C_y = C[:, 1::2]
    D_x = np.array([np.sum(C_x, axis=1)]).T
    D_y = np.array([np.sum(C_y, axis=1)]).T
    D = np.concatenate((D_x, D_y), axis=1)
    prey_alignment = parameter_alignment_strength / prey_to_prey_degreeMatrix * D
    prey_alignment = np.nan_to_num(prey_alignment)


    # total force of prey
    prey_total_force = prey_flight+prey_alignment
    norm_prey = mf.get_norm_of_vector(prey_total_force)
    unit_prey = mf.get_unit_of_vector(prey_total_force)
    norm_prey[norm_prey>maxVelocity_prey]=maxVelocity_prey
    delta_prey = norm_prey * unit_prey

    # predator maneuverability

    current_prey_position = copy.deepcopy(prey_position[:,:,current_step])
    current_predator_position = copy.deepcopy(predator_position[:,:,current_step])


    [predator_to_predator_distanceMatrix, predator_to_predator_adjacencyMatrix, predator_to_predator_degreeMatrix] = mf.get_adjacency_matrix(
        current_predator_position, current_predator_position, sensingRange_predator)

    # repulsion force between predators
    A = np.tile(current_predator_position, (1, M))
    BB = current_predator_position.reshape(1, -1)
    B = np.tile(np.array(BB), (M, 1))

    H = np.zeros((M, 2 * M))
    H[:, ::2] = predator_to_predator_adjacencyMatrix
    H[:, 1::2] = predator_to_predator_adjacencyMatrix

    Z = np.zeros((M, 2 * M))
    Z[:, ::2] = predator_to_predator_distanceMatrix ** 2
    Z[:, 1::2] = predator_to_predator_distanceMatrix ** 2

    Z[Z==0]=10**100

    C = (H * (A - B)) / Z
    C = np.nan_to_num(C)

    C_x = C[:, ::2]
    C_y = C[:, 1::2]
    D_x = np.array([np.sum(C_x, axis=1)]).T
    D_y = np.array([np.sum(C_y, axis=1)]).T
    D = np.concatenate((D_x, D_y), axis=1)
    predator_repulsion = parameter_repulsion_strength / predator_to_predator_degreeMatrix * D
    predator_repulsion = np.nan_to_num(predator_repulsion)



    # herding control strategy
    # 1 push edge prey
    [predator_to_prey_distanceMatrix, predator_to_prey_adjacencyMatrix,
     predator_to_prey_degreeMatrix] = mf.get_adjacency_matrix(
        current_predator_position, current_prey_position, sensingRange_predator)

    DD = mf.get_norm_of_vector(current_prey_position-np.tile(target_position, (N, 1)))
    EE = np.tile(np.array(DD).T, (M, 1))
    AA = predator_to_prey_adjacencyMatrix* EE
    idx = np.array([np.argmax(AA, axis=1)]).T
    norm_dist=np.array([np.amax(AA, axis=1)]).T   # the distance between the prey and the target[0,0]
    edge_prey = current_prey_position[idx,:]  # the selected edge prey
    edge_prey =edge_prey.reshape(M,2)
    CC = predator_to_prey_degreeMatrix
    CC[CC>1]=1
    virtual_points = mf.get_unit_of_vector(edge_prey-np.tile(target_position, (M, 1)))*pushRange + edge_prey
    predator_push_edge_prey = parameter_push_strength*CC*(virtual_points-current_predator_position) # los pursue

    # total force of predator
    predator_total_force = predator_repulsion + predator_push_edge_prey
    norm_predator = mf.get_norm_of_vector(predator_total_force)
    unit_predator = mf.get_unit_of_vector(predator_total_force)
    norm_predator[norm_predator>maxVelocity_predator]=maxVelocity_predator
    delta_predator= norm_predator * unit_predator

    # update position
    prey_position[:,:, current_step+1]=prey_position[:,:,current_step]+delta_prey*deltaT
    predator_position[:,:,current_step+1]=predator_position[:,:,current_step]+delta_predator* deltaT

    current_step += 1

    print("current_step is", current_step)

p1 = time.process_time()
spend = p1 - p0
print("process_time()用时：{}s".format(spend))


# figure plot



for i in range(N):

    x_data = prey_position[i,0,:]
    y_data = prey_position[i,1,:]
    plt.plot(x_data,y_data,ls='-', c='g', lw=0.1)
for i in range(M):

    plt.plot(predator_position[i,0,:],predator_position[i,1,:],ls='-', c='r', lw=0.1)

data_x_end =prey_position[:,0,-1]
data_y_end= prey_position[:,1,-1]

data_x_0 =prey_position[:,0,0]
data_y_0= prey_position[:,1,0]

plt.scatter(data_x_0,data_y_0,color='green',marker='.')# 画散点图
plt.scatter(data_x_end,data_y_end,color='g',marker='.')# 画散点图
plt.scatter(predator_position[:,0,0],predator_position[:,1,0],color='red',marker='.')
plt.scatter(predator_position[:,0,-1],predator_position[:,1,-1],color='red',marker='.')
plt.show()
plt.axis('equal')

# draw the sheep shed







