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
    #This approach is using Dynamic Programming

    value = [[[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))],
             [[999 for col in range(len(grid[0]))] for row in range(len(grid))]]
    
    #for the policy
    policy = [[[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
             [[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
             [[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
             [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]]
    
    change = True    
    while change:
        change = False
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                for orient in range(len(forward)):
                    
                    if row == goal[0] and col == goal[1]: #orient could be anything    
                        if value[orient][row][col] > 0:
                            value[orient][row][col] = 0 #zero cost for goal in any orientation
                            change = True
                            #policy
                            policy[orient][row][col] = '*'
                            
                    elif grid[row][col] == 0:
                        for act in range(len(action)):
                            mv_dir = action[act] + orient
                            mv_dir = mv_dir % len(forward) #normalize to between 0 and len(forward)-1
                            row2 = row + forward[mv_dir][0]
                            col2 = col + forward[mv_dir][1]
                            
                            if (row2 >= 0 and row2 < len(grid) and col2 >= 0 and col2 < len(grid[0]) and grid[row2][col2] == 0):
                                v2 = value[mv_dir][row2][col2] + cost[act]
                                if v2 < value[orient][row][col]:
                                    value[orient][row][col] = v2
                                    #for policy
                                    policy[orient][row][col] = act #action_name[act]
                                    change = True
    
                                    
    policy2D = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    policy2D[goal[0]][goal[1]] = '*'
    
    #Run the policy to find the path from start
    cur_loc = init
    while (cur_loc[0] != goal[0]) or (cur_loc[1] != goal[1]):
        row,col,orient = cur_loc
        act = policy[orient][row][col]
        policy2D[row][col] = action_name[act]
        
        orient2 = orient + action[act]
        orient2 = orient2 % len(forward)
        row2 = row + forward[orient2][0]
        col2 = col + forward[orient2][1]
        cur_loc = [row2,col2,orient2]
    
    pth_len = value[init[2]][init[0]][init[1]]
    return [pth_len,policy2D] #[value, policy] #policy2D
 
pth_len, pol_2d = optimum_policy2D(grid,init,goal,cost)
print("Path Length is: ", pth_len)
print()
for i in range(len(pol_2d)):
    print(pol_2d[i])

#val_tmp, pol_tmp = optimum_policy2D(grid,init,goal,cost)
#for i in range(len(pol_tmp)):
#    print("\ni Value is:", i)
#    for j in range(len(grid)):
#        print(val_tmp[i][j])
#    print()
#    for j in range(len(grid)):
#        print(pol_tmp[i][j])        
    
        
#def optimum_policy2D(grid,init,goal,cost):
#    # ----------------------------------------
#    # This is based on uniform cost search approach
#    # ----------------------------------------
#    #closed is same size as the grid used to check whether a node has been expanded/checked
#    #0 means it's not checked and 1 means it has been checked
#    closed = [[[0 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[0 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[0 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[0 for row in range(len(grid[0]))] for col in range(len(grid))]]
#    closed[init[2]][init[0]][init[1]] = 1 #initialize the starting location as checked
#
#    row = init[0]
#    col = init[1]
#    orient = init[2]
#    g = 0
#
#    fringe = []
#    fringe.append([g, row, col, orient])
#
#    found = False  # flag that is set when search is complete
#    resign = False # flag set if we can't find/expand
#
#   
#
#    #find the path
#    actions3D = [[[-1 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[-1 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[-1 for row in range(len(grid[0]))] for col in range(len(grid))],
#              [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]]
#    
#    
#    while not found and not resign:
#        if len(fringe) == 0:
#            resign = True
#            return_val = 'fail'
#            break
#        else:
#            fringe.sort() #sorts by the first value in the list element else use the key parameter
#            fringe.reverse()
#            next = fringe.pop()
#            row = next[1]
#            col = next[2]
#            orient = next[3]
#            g = next[0]
#            
#            if row == goal[0] and col == goal[1]:
#                found = True
#                return_val = g #path length to the goal
#                goal_end = [row,col,orient]
#                break
#            else:
#                for act in range(len(action)):
#                    orient2 = action[act] + orient
#                    orient2 = orient2 % len(forward) #normalize to between 0 and len(forward)-1
#                    row2 = row + forward[orient2][0]
#                    col2 = col + forward[orient2][1]                    
#
#                    if row2 >= 0 and row2 < len(grid) and col2 >=0 and col2 < len(grid[0]):
#                        if closed[orient2][row2][col2] == 0 and grid[row2][col2] == 0:
#                            g2 = g + cost[act]
#                            fringe.append([g2, row2, col2, orient2])
#                            closed[orient2][row2][col2] = 1 #mark it checked
#                            #actions
#                            actions3D[orient2][row2][col2] = act #[row2,col2] came from [row,col] using direction act
#    
#    #print("1 down")
#    
#    if return_val != 'fail':
#        policy2D = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
#        curr_loc = goal_end
#        policy2D[curr_loc[0]][curr_loc[1]] = '*'
#        
#        while(curr_loc[0] != init[0] or curr_loc[1] != init[1] or curr_loc[2] != init[2]):
#            print(curr_loc)
#            row,col,orient = curr_loc
#            row_prev = row - forward[orient][0]
#            col_prev = col - forward[orient][1] 
#
#            act = actions3D[orient][row][col]
#            print(action_name[act])
#            orient_prev = orient - action[act]
#            orient_prev = orient_prev % len(forward) #normalize to between 0 and len(forward)-1
#            
#            #row_prev = row - forward[orient_prev][0]
#            #col_prev = col - forward[orient_prev][1]                    
#
#            policy2D[row_prev][col_prev] = action_name[act]
#            curr_loc = [row_prev,col_prev,orient_prev]
#            print(curr_loc)
#            #input("Press Enter to continue...")
#            
#        return_val = [return_val] + [policy2D] + [actions3D]
#    #return_val = [return_val] + [actions3D]
#    
#    return return_val
#    
#pth_len, pol_2d,act_3d = optimum_policy2D(grid,init,goal,cost)
#print("Path Length is: ", pth_len)
#print()
#for i in range(len(pol_2d)):
#    print(pol_2d[i])
#
#print()
#for i in range(len(act_3d)):
#    print()
#    for j in range(len(act_3d[0])):
#        print(act_3d[i][j])

