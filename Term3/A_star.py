#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:18:56 2017

@author: amit
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:01:47 2017

@author: amit_p
"""

##### AMIT'S SOLUTION ###########
## ----------
## User Instructions:
## 
## Define a function, search() that returns a list
## in the form of [optimal path length, row, col]. For
## the grid shown below, your function should output
## [11, 4, 5].
##
## If there is no valid path from the start point
## to the goal, your function should return the string
## 'fail'
## ----------
#
## Grid format:
##   0 = Navigable space
##   1 = Occupied space
##grid = [[0, 0, 1, 0, 0, 0],
##        [0, 0, 1, 0, 0, 0],
##        [0, 0, 0, 0, 1, 0],
##        [0, 0, 1, 1, 1, 0],
##        [0, 0, 0, 0, 1, 0]]
#
#grid = [[0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 1, 0, 1, 0],
#        [0, 0, 1, 0, 1, 0],
#        [0, 0, 1, 0, 1, 0]]
#
#init = [0, 0]
#goal = [len(grid)-1, len(grid[0])-1]
#cost = 1
#delta = [[-1, 0], # go up
#         [ 0,-1], # go left
#         [ 1, 0], # go down
#         [ 0, 1]] # go right
#         
#delta_name = ['^', '<', 'v', '>']
#
##defined by Amit
#def l1_gtrthneql_l2(list1, list2):
#    
#    assert (len(list1) == len(list2)), "Not Same Sized Lists!"
#    compare = True
#    for i in range(len(list1)):
#        if list1[i] < list2[i]:
#            compare = False
#            return compare #need only one element to fail
#    return compare #return True
##end amit function
#
#def search(grid,init,goal,cost):
#    # ----------------------------------------
#    # insert code here
#    current_pos = [0] + init #0 is the optimal path length for the starting position
#    open_list = []
#    expanded_points = []
#    
#    y_len = len(grid)
#    x_len = len(grid[0])
#    
#    itr = 0
#    while (current_pos[1:] != goal):
#        itr += 1
#        #print(itr, current_pos)
#        #print(expanded_points)
#        go_up = [current_pos[0]+cost, current_pos[1]+delta[0][0], current_pos[2]+delta[0][1]]
#        go_left = [current_pos[0]+cost, current_pos[1]+delta[1][0], current_pos[2]+delta[1][1]]
#        go_down = [current_pos[0]+cost, current_pos[1]+delta[2][0], current_pos[2]+delta[2][1]]
#        go_right = [current_pos[0]+cost, current_pos[1]+delta[3][0], current_pos[2]+delta[3][1]]
#        
#        expanded_points.append(current_pos[1:])
#
#        open_list_tmp = list(map(lambda x: x[1:],open_list))
#        #only add points to open_list that and not in the expanded list
#        if (l1_gtrthneql_l2(go_up[1:],[0,0]) and l1_gtrthneql_l2([y_len-1,x_len-1],go_up[1:]) and (grid[go_up[1]][go_up[2]] == 0) and (go_up[1:] not in expanded_points) and (go_up[1:] not in open_list_tmp)):
#            open_list.append(go_up)
#        if (l1_gtrthneql_l2(go_left[1:],[0,0]) and l1_gtrthneql_l2([y_len-1,x_len-1],go_left[1:]) and (grid[go_left[1]][go_left[2]] == 0) and (go_left[1:] not in expanded_points) and (go_left[1:] not in open_list_tmp)):
#            open_list.append(go_left)
#        if (l1_gtrthneql_l2(go_down[1:],[0,0]) and l1_gtrthneql_l2([y_len-1,x_len-1],go_down[1:]) and (grid[go_down[1]][go_down[2]] == 0) and (go_down[1:] not in expanded_points) and (go_down[1:] not in open_list_tmp)):
#            open_list.append(go_down)
#        if (l1_gtrthneql_l2(go_right[1:],[0,0]) and l1_gtrthneql_l2([y_len-1,x_len-1],go_right[1:]) and (grid[go_right[1]][go_right[2]] == 0) and (go_right[1:] not in expanded_points) and (go_right[1:] not in open_list_tmp)):
#            open_list.append(go_right)
#            
#        
#        open_list.sort(key=lambda pos: pos[0], reverse=True) #sort by the path length and sort in reverse
#        #print(open_list)
#
#        if(len(open_list) == 0):
#            return 'fail'
#        else:                      
#            current_pos = open_list.pop() #pop the one with lowest path length
#        
#        #input("Press Enter to continue...")    
#    # ----------------------------------------
#    
#    path = current_pos[0] #this is the total path length/cost
#    return path
#    
#path_travelled = search(grid,init,goal,cost)
#print(path_travelled)
####### END AMIT'S SOLUTION######


####### SEBASTIAN'S SOLUTION ###############
# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------
#
# Modify the the search function so that it returns
# a shortest path as follows:
# 
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left, 
# up, and down motions. Note that the 'v' should be 
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# The expanded grid shows, for each element, the count when
# it was expanded or -1 if the element was never expanded.
# 
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

#### NOTE: X=ROWS AND Y=COLUMNS IN THIS CASE (COUNTERINTUITIVE!)

#grid = [[0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 1, 0, 1, 0],
#        [0, 0, 1, 0, 1, 0],
#        [0, 0, 1, 0, 1, 0]]

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
        
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

#for basic search, make heuristic be all zeros
#heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]

                
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost,heuristic):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    #closed is same size as the grid used to check whether a node has been expanded/checked
    #0 means it's not checked and 1 means it has been checked
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1 #initialize the starting location as checked

    x = init[0]
    y = init[1]
    g = 0
    f = g + heuristic[goal[0]][goal[1]]

    open = [[g, x, y, f]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find/expand
   
    #expand is the same size as the grid
    #-1 means not expanded
    #else it means the point is expanded/checked at that location
    expand = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    count = 0
    #
    #find the path
    actions = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]
    
    
    while not found and not resign:
        if len(open) == 0:
            resign = True
            return_val = ['fail', expand]
            break
        else:
            open.sort(key=lambda pos: pos[3]) #sorts by the 4 value in the list element
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            
            #
            expand[x][y] = count
            count += 1
            #
            
            if x == goal[0] and y == goal[1]:
                found = True
                return_val = [g,expand] #path length to the goal
                break
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            open.append([g2, x2, y2, f2])
                            closed[x2][y2] = 1 #mark it checked
                            #actions
                            actions[x2][y2] = i #[x2,y2] came from [x,y] using direction i
                            
    if return_val[0] != 'fail':
        path_grid = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
        curr_loc = tuple(goal)
        path_grid[curr_loc[0]][curr_loc[1]] = '*'
        
        while(curr_loc != tuple(init)):
            x,y = curr_loc
            direction = actions[x][y]
            x_prev = x - delta[direction][0]
            y_prev = y - delta[direction][1]
            path_grid[x_prev][y_prev] = delta_name[direction]
            curr_loc = (x_prev,y_prev)            
            #input("Press Enter to continue...")
            
        return_val = return_val + [path_grid]
    
    return return_val
        
path_len, expand, path_grid = search(grid,init,goal,cost,heuristic)
print("Shortest path length:", path_len)
print("\nExpanded Grid")
for i in range(len(expand)):
    print(expand[i])
print("\nShortest Path")
for i in range(len(path_grid)):
    print(path_grid[i])
    
############ END SEBASTIAN'S SOLUTION##########