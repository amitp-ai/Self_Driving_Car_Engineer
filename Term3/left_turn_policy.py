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

cost = [2, 1, 20] #[2, 1, 20] # cost has 3 values, corresponding to making 
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
                if row == goal[0] and col == goal[1]: #orient could be anything
                    for orient in range(len(forward)):
                        if value[orient][row][col] > 0:
                            value[orient][row][col] = 0 #zero cost for goal in any orientation
                            change = True
                            #policy
                            policy[orient][row][col] = '*'
                            
                elif grid[row][col] == 0:
                    for orient in range(len(forward)):
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
#    
#    #The below approach is based on uniform cost search algorithm
#    
#    # ----------------------------------------
#    # modify code below
#    # ----------------------------------------
#    #closed is same size as the grid used to check whether a node has been expanded/checked
#    #0 means it's not checked and 1 means it has been checked
#    
#    #The search space is really 3-dimensional: x,y,direction
#    closed = [[[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))]]
#              
#    closed[init[2]][init[0]][init[1]] = 1 #initialize the starting location as checked
#
#    row = init[0]
#    col = init[1]
#    drct = init[2]
#    g = 0
#
#    fringe = [[g, row, col, drct]]
#
#    found = False  # flag that is set when search is complete
#    resign = False # flag set if we can't find/expand
#
#    #expand is the same size as the grid
#    #-1 means not expanded
#    #else it means the point is expanded/checked at that location
#    #The search space is really 3-dimensional: x,y,direction
#    expand = [[[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#              [[0 for col in range(len(grid[0]))] for row in range(len(grid))]]
#
#    count = 0
#    #
#    #find the path
#    #The action space is really 3-dimensional: x,y,action
#    action_list = [[[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#                   [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#                   [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
#                   [[0 for col in range(len(grid[0]))] for row in range(len(grid))]]
#    
#    
#    dir_list = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
#    
#    while not found and not resign:
#        if len(fringe) == 0:
#            resign = True
#            return_val = ['fail', expand]
#            break
#        else:
#            fringe.sort() #sorts by the first value in the list element (else use the key parameter)
#            fringe.reverse()
#            
#            #print(fringe)
#            
#            next = fringe.pop()
#            row = next[1]
#            col = next[2]
#            drct = next[3]
#            g = next[0]
#            
#            #print(next)
#            #input("Press Enter to continue...")    
#
#            #
#            expand[drct][row][col] = count
#            count += 1
#            #
#            dir_list[row][col] = drct
#            if row == goal[0] and col == goal[1]:
#                found = True
#                return_val = [g,expand] #path length to the goal
#                break
#            else:
#                closed[drct][row][col] = 1 #unlike the 2D case, in this case mark it checked only when it is poped out of list not when on the fringe
#
#                for act in range(len(action)):
#                    drct2 = drct + action[act]
#                    drct2 = drct2 % len(forward) #normalize to between 0 and len(forward)-1
#                    
#                    row2 = row + forward[drct2][0]
#                    col2 = col + forward[drct2][1]
#                                                            
#                    if row2 >= 0 and row2 < len(grid) and col2 >=0 and col2 < len(grid[0]):
#                        if closed[drct2][row2][col2] == 0 and grid[row2][col2] == 0:
#                            g2 = g + cost[act]
#                            fringe.append([g2, row2, col2, drct2])
#                            action_list
##                            if g2 < gmin:
##                                gmin = g2
##                                action_list[row][col] = act #[row2,col2] came from [row,col] using action act
##                                dir_list[row][col] = drct
#                            
#    if return_val[0] != 'fail':
#        path_grid = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
#        path_grid[goal[0]][goal[1]] = '*'
#        
#        curr_loc = tuple(init[0:2])
#        dir_temp = init[2]
#        
#        while(curr_loc != tuple(goal)):
#            row,col = curr_loc
#            #dir_temp = dir_list[row][col]
#            act_temp = action_list[row][col]
#            
#            dir_nxt = dir_temp + action[act_temp]
#            dir_nxt = dir_nxt % len(forward) #normalize to between 0 and len(forward)-1
#            
#            row_nxt = row + forward[dir_nxt][0]
#            col_nxt = col + forward[dir_nxt][1]
#            path_grid[row][col] = action_name[act_temp]
#            
#            curr_loc = (row_nxt,col_nxt)   
#            dir_temp = dir_nxt
#            #input("Press Enter to continue...")
#            
#        return_val = return_val + [path_grid]
#    
#    return return_val
#    
#plen, pexp, pgrid = optimum_policy2D(grid,init,goal,cost)
#
#for i in range(len(pgrid)):
#    print(pgrid[i])
    
