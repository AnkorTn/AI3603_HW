import DR20API
import numpy as np
# the minimun heap
from heapq import *
import matplotlib.pyplot as plt
### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

def general_cost(current_node, x, y):
    if not current_node.parent:
        return current_node.g_cost + 1
    v1 = [x-current_node.current_pos[0], y-current_node.current_pos[1]]
    v2 = [x-current_node.parent.current_pos[0], y-current_node.parent.current_pos[1]]
    v3 = v1[0]*v2[1] - v1[1]*v2[0]
    if(v3 == 0):
        return current_node.g_cost + 1
    return current_node.g_cost + 3

# The heuristics_cost is cauculated by Manhantan Distance
def heuristics_cost(current_node, goal_node):
    # Euclidean Distance
    # return ((current_node[1]-goal_node[1])**2 + (current_node[0]-goal_node[0])**2)**0.5

    # Manhantan Distance
    return np.sum(np.abs(current_node-goal_node))

def total_cost(current_node, goal_node):
    return heuristics_cost(current_node, goal_node)

class Node:
    def __init__(self, current_pos, parent, cost, g_cost):
        self.current_pos = current_pos
        self.parent = parent
        self.cost = cost
        self.g_cost = g_cost
    
    # define how to compare two nodes
    def __lt__(self, other):
        return self.cost < other.cost

###  END CODE HERE  ###

def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###

    # define the open_set and the close_set
    open_set, close_set = [], set()

    # record the step from start to the current (general_cost)
    # step = 0

    # push the start_node into the minimun heap
    heappush(open_set, Node(current_pos, None, heuristics_cost(current_pos,goal_pos), 0))


    # pop node from the open_set
    while (open_set):
        node = heappop(open_set)

        # update the step
        # step += 1

        # if node locates in the goal, we can stop and find a path
        if (reach_goal(node.current_pos, goal_pos)):
            break


        # direct = [(-1,1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]

        # if there is no obstacle and the next_point is still on the map, push it into the heap
        for next_x, next_y in [(node.current_pos[0]-1,node.current_pos[1]), (node.current_pos[0]+1,node.current_pos[1]), (node.current_pos[0],node.current_pos[1]-1), (node.current_pos[0],node.current_pos[1]+1)]:
            # flag = True
            # for n_x, n_y in direct:
                # if(current_map[next_x+n_x][next_y+n_y]==1):
                    # flag = False
                    # break
            # if(flag == False):
                # continue
            if (0 < next_x < 119 and 0 < next_y < 119 and current_map[next_x][next_y]==0 and (next_x, next_y) not in close_set):
                g_cost = general_cost(node,next_x,next_y)
                heappush(open_set, Node(np.array([next_x, next_y]), node, heuristics_cost(np.array([next_x, next_y]), goal_pos) + g_cost, g_cost))
                # nodes in close_set will not be searched again
                close_set.add((next_x,next_y))

    path = [[node.current_pos[0], node.current_pos[1]]]

    # find the whole path through the parent_pos
    while (node.parent):
        path.append([node.parent.current_pos[0],node.parent.current_pos[1]])
        node = node.parent

    path = path[::-1]
    # print(path)
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    ###  END CODE HERE  ###
    return path

def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###

    if (abs(current_pos[0] - goal_pos[0])<=5 and abs(current_pos[1] - goal_pos[1])<=5):
        is_reached = True
    else:
        is_reached = False

    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()
    # plt.show()