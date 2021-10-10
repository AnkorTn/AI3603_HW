import DR20API
import numpy as np
# the minimun heap
from heapq import *
import matplotlib.pyplot as plt
import math

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.


def general_cost(current_node):
    return current_node.g_cost + 0.2

# def general_cost2(current_node):
#     return current_node.g_cost + 2**0.5


# The heuristics_cost is cauculated by Manhantan Distance
def heuristics_cost(current_node, goal_node):
    # Euclidean Distance
    return ((current_node[1]-goal_node[1])**2 + (current_node[0]-goal_node[0])**2)**0.5

# def total_cost(current_node, goal_node):
#     return heuristics_cost(current_node, goal_node)

class Node:
    def __init__(self, current_pos, angle, layer, parent, cost, g_cost):
        self.x = current_pos[0]
        self.y = current_pos[1]
        self.angle = angle
        self.parent = parent
        self.cost = cost
        self.g_cost = g_cost

        # define the deepest search layer
        self.layer = layer
    
    # define how to compare two nodes
    def __lt__(self, other):
        return self.cost < other.cost

###  END CODE HERE  ###

# def Hybrid_A_star(current_map, current_pos, goal_pos):
def Hybrid_A_star(current_map, current_pos, goal_pos, angle, total):
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
    # print(total)
    ### START CODE HERE ###

    # define the open_set and the close_set
    open_set, close_set = [], set()

    # push the start_node into the minimun heap
    heappush(open_set, Node(current_pos, angle, 0, None, heuristics_cost(current_pos,goal_pos), 0))

    # pop node from the open_set
    while (open_set):
        node = heappop(open_set)

        # if node locates in the goal, we can stop and find a path
        current_pos = np.array([node.x, node.y])
        # if (node.layer > 200 or reach_goal(current_pos, goal_pos)):
        if (reach_goal(current_pos, goal_pos)):
            break

        if(current_pos[0] >=110 or current_pos[1] >=110):
            continue

        # possible steering angle
        steering = [-0.03*math.pi, 0, 0.03*math.pi]
        cost_steering = [0.2, 0, 0.2]

        # possible steering angle
        # steering = [-0.4*math.pi, -0.2*math.pi,  0,  0.2*math.pi, 0.4*math.pi]
        # cost_steering = [2, 1, 0, 1, 2]

        # current position of the robot
        x, y = node.x, node.y
        
        # if there is no obstacle and the next_point is still on the map, push it into the heap
        for i, angle in enumerate(steering):
            # cost for turn
            steering_cost = cost_steering[i]

            # new angle
            angle += node.angle

            # calculate the next position of the robot according to the turn
            # cx, cy represent the continue position of the robot
            # dx, dy represent the discrete position of the robot
            next_cx, next_cy = round(x + 0.25*math.cos(angle),2), round(y + 0.25  *math.sin(angle),2)
            next_dx, next_dy = int(next_cx), int(next_cy)

            # judge weather near the obstacle
            # 这里现在flag直接用来作为cost记录
            # flag = current_map[next_dx][next_dy] + current_map[next_dx+1][next_dy] + current_map[next_dx-1][next_dy] + current_map[next_dx][next_dy+1] + current_map[next_dx][next_dy-1] \
            flag = current_map[next_dx+2][next_dy+1] + current_map[next_dx+2][next_dy-1]   \
                   + current_map[next_dx+2][next_dy]  + current_map[next_dx][next_dy+2]  \
                   + current_map[next_dx+1][next_dy+2] + current_map[next_dx-1][next_dy+2]  \
                   + current_map[next_dx+2][next_dy+2] + current_map[next_dx-1][next_dy+1] \
                   + current_map[next_dx-1][next_dy] + current_map[next_dx-1][next_dy-1] \
                   + current_map[next_dx][next_dy-1] + current_map[next_dx+1][next_dy-1]
            
            # New definition of flag
            flag *= 2

            flag +=  current_map[next_dx+3][next_dy+1] + current_map[next_dx+3][next_dy-1]   \
                   + current_map[next_dx+3][next_dy]  + current_map[next_dx][next_dy+3]  \
                   + current_map[next_dx+1][next_dy+3] + current_map[next_dx-1][next_dy+3]  \
                   + current_map[next_dx+2][next_dy+3] + current_map[next_dx-1][next_dy-2] \
                   + current_map[next_dx-2][next_dy] + current_map[next_dx-2][next_dy-1] \
                   + current_map[next_dx][next_dy-2] + current_map[next_dx+1][next_dy-2] \
                   + current_map[next_dx+2][next_dy-2] + current_map[next_dx+3][next_dy-2] \
                   + current_map[next_dx+3][next_dy+2] + current_map[next_dx+3][next_dy+3] \
                   + current_map[next_dx-2][next_dy-2] + current_map[next_dx-2][next_dy+1] \
                   + current_map[next_dx-2][next_dy+2] + current_map[next_dx-2][next_dy+3]

            flag *= 0.1

            obs = current_map[next_dx][next_dy] + current_map[next_dx+1][next_dy] + current_map[next_dx][next_dy+1] + current_map[next_dx+1][next_dy+1]

            # 然后flag==0就可以排除掉了，变成该位置不为障碍
            if (0 < next_cx < 110 and 0 < next_cy < 110 and not obs and (next_cx, next_cy) not in close_set):
                heappush(open_set, Node(np.array([next_cx, next_cy]), angle, node.layer+1, node, heuristics_cost(np.array([next_cx, next_cy]), goal_pos) + flag + general_cost(node) + steering_cost, general_cost(node) + steering_cost))
                # heappush(open_set, Node(np.array([next_cx, next_cy]), angle, node.layer+1, node, heuristics_cost(np.array([next_cx, next_cy]), goal_pos) + flag + general_cost(node) + steering_cost, general_cost(node) + steering_cost))
                # nodes in close_set will not be searched again
                close_set.add((next_cx,next_cy))

    path = [[node.x, node.y]]

    # find the whole path through the parent_pos
    while (node.parent):
        path.append([node.parent.x,node.parent.y])
        node = node.parent

    path = path[::-1]
    # path[0][1] += 1
    # print(path)


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
    # plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    # plt.show()
    plt.savefig(r"D:\College\5\AI\AI3603_HW1\result4\filename{}.png".format(total))
    # print(path)
    # print(annngle)
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

    if (abs(current_pos[0] - goal_pos[0])<=10 and abs(current_pos[1] - goal_pos[1])<=10):
        is_reached = True
    else:
        is_reached = False

    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()
    total = 0

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        total += 1
        # Get current orientation of the robot.
        current_ori = controller.get_robot_ori()
        path = Hybrid_A_star(current_map, current_pos, goal_pos, current_ori, total)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        # current_pos = controller.get_robot_pos()
        current_pos = controller.get_robot_pos()
        # current_pos = np.array(path[-1])
        # This current_pos2 is to avoid the 
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()

###  END CODE HERE  ###