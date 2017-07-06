#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:35:48 2017

@author: amit
"""

# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------
         
def optimum_policy2D(grid,init,goal,cost):

    value = [[[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))]]
    #for the policy
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    
    #direct values are 0,1,2,3
    direction = [[0 for col in range(len(grid[0]))] for row in range(len(grid))] #initial direction is 0 for everyone
    direction[init[0]][init[1]] = init[2] #starting direction is fixed

    change = True    
    while change:
        change = False
        
        for row in range(len(value[0])):
            for col in range(len(value[0][0])):
                if row == goal[0] and col == goal[1]:
                    for act in range(len(action)):
                        if value[act][row][col] > 0:
                            value[act][row][col] = 0 #zero cost for goal in any orientation
                            change = True
                            #policy
                            policy[row][col] = '*'
                            
                elif grid[row][col] == 0:
                    for act in range(len(value)):
                        mv_dir = action[act] + direction[row][col]
                        mv_dir = mv_dir % len(forward) #normalize to between 0 and len(forward)-1
                        row2 = row + forward[mv_dir][0]
                        col2 = col + forward[mv_dir][1]
                        
                        v2 = value[][row][col]
                    
                    
                
    

    
    return policy2D
    
