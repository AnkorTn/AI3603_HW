import DR20API
import numpy as np
# the minimun heap
from heapq import *
import matplotlib.pyplot as plt

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

# def general_cost(current_node, start_node):
#     return 0

def general_cost1(current_node):
    return current_node.g_cost + 1

def general_cost2(current_node):
    return current_node.g_cost + 2**0.5


# The heuristics_cost is cauculated by Manhantan Distance
def heuristics_cost(current_node, goal_node):
    # Euclidean Distance
    return ((current_node[1]-goal_node[1])**2 + (current_node[0]-goal_node[0])**2)**0.5

    # Manhantan Distance
    # return np.sum(np.abs(current_node-goal_node))

# def total_cost(current_node, goal_node):
#     return heuristics_cost(current_node, goal_node)

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

def Improved_A_star(current_map, current_pos, goal_pos):
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

    # push the start_node into the minimun heap
    heappush(open_set, Node(current_pos, None, heuristics_cost(current_pos,goal_pos), 0))

    # pop node from the open_set
    while (open_set):
        node = heappop(open_set)

        # if node locates in the goal, we can stop and find a path
        if (reach_goal(node.current_pos, goal_pos)):
            break

        # up, down, left, right
        direct1 = [(node.current_pos[0]-1,node.current_pos[1]), (node.current_pos[0]+1,node.current_pos[1]), (node.current_pos[0],node.current_pos[1]-1), (node.current_pos[0],node.current_pos[1]+1)]
        # NW, NE, SW, SE
        direct2 = [(node.current_pos[0]-1,node.current_pos[1]-1), (node.current_pos[0]+1,node.current_pos[1]+1), (node.current_pos[0]+1,node.current_pos[1]-1), (node.current_pos[0]-1,node.current_pos[1]+1)]
        
        # if there is no obstacle and the next_point is still on the map, push it into the heap
        for next_x, next_y in direct1:
            # cost for turn
            turn_cost = 0

            # judge weather near the obstacle
            flag = current_map[next_x][next_y] + current_map[next_x+1][next_y] + current_map[next_x-1][next_y] + current_map[next_x][next_y+1] + current_map[next_x][next_y-1] \
                   + current_map[next_x+1][next_y+1] + current_map[next_x+1][next_y-1] + current_map[next_x-1][next_y+1] + current_map[next_x-1][next_y-1]
            
            if (0 < next_x < 119 and 0 < next_y < 119 and flag==0 and (next_x, next_y) not in close_set):
                # judge three points collinear
                if node.parent and (next_y - node.parent.current_pos[1])*(node.current_pos[0] - node.parent.current_pos[0]) - (node.current_pos[1] - node.parent.current_pos[1])*(next_x - node.parent.current_pos[0]) != 0:
                    turn_cost += 2
                heappush(open_set, Node(np.array([next_x, next_y]), node, heuristics_cost(np.array([next_x, next_y]), goal_pos) + general_cost1(node) + turn_cost, general_cost1(node) + turn_cost))
                # nodes in close_set will not be searched again
                close_set.add((next_x,next_y))
        for next_x, next_y in direct2:
            # cost for turn
            turn_cost = 0

            # judge weather near the obstacle
            flag = current_map[next_x][next_y] + current_map[next_x+1][next_y] + current_map[next_x-1][next_y] + current_map[next_x][next_y+1] + current_map[next_x][next_y-1] \
                   + current_map[next_x+1][next_y+1] + current_map[next_x+1][next_y-1] + current_map[next_x-1][next_y+1] + current_map[next_x-1][next_y-1]

            if (0 < next_x < 119 and 0 < next_y < 119 and flag==0 and (next_x, next_y) not in close_set):
                # judge three points collinear
                if node.parent and (next_y - node.parent.current_pos[1])*(node.current_pos[0] - node.parent.current_pos[0]) - (node.current_pos[1] - node.parent.current_pos[1])*(next_x - node.parent.current_pos[0]) != 0:
                    turn_cost += 2
                heappush(open_set, Node(np.array([next_x, next_y]), node, heuristics_cost(np.array([next_x, next_y]), goal_pos) + general_cost2(node) + turn_cost, general_cost2(node) + turn_cost))
                # nodes in close_set will not be searched again
                close_set.add((next_x,next_y))


    path = [[node.current_pos[0], node.current_pos[1]]]

    # find the whole path through the parent_pos
    while (node.parent):
        path.append([node.parent.current_pos[0],node.parent.current_pos[1]])
        node = node.parent

    path = path[::-1]
    print(path)


    # Visualize the map and path.
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

    if (abs(current_pos[0] - goal_pos[0])<=8 and abs(current_pos[1] - goal_pos[1])<=8):
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
        path = Improved_A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()