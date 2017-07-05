#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 20:00:07 2017

@author: amit
"""

# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------
# Write a function optimum_policy that returns
# a grid which shows the optimum policy for robot
# motion. This means there should be an optimum
# direction associated with each navigable cell from
# which the goal can be reached.
# 
# Unnavigable cells as well as cells from which 
# the goal cannot be reached should have a string 
# containing a single space (' '), as shown in the 
# previous video. The goal cell should have '*'.
# ----------


grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------

    #find the values and policy
    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
    #for the policy
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    
    change = True
    while change:
        change = False
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if row == goal[0] and col == goal[1]:
                    if value[row][col] > 0:
                        value[row][col] = 0
                        change = True
                        #for policy
                        policy[row][col] = '*'
                                
                elif grid[row][col] == 0:
                    for a in range(len(delta)):
                        row2 = row + delta[a][0]
                        col2 = col + delta[a][1]
                        
                        if (row2 >= 0 and row2 < len(grid) and col2 >= 0 and col2 < len(grid[0]) and grid[row2][col2] == 0):
                            v2 = value[row2][col2] + cost
                            if v2 < value[row][col]:
                                value[row][col] = v2
                                change = True
                                #for policy
                                policy[row][col] = delta_name[a]

    return [value,policy]
                
value, policy = compute_value(grid,goal,cost)
for i in range(len(value)):
    print(value[i])
print()
for i in range(len(policy)):
    print(policy[i])    
    
    
    