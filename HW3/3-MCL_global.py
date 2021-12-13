import DR20API
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, pi
import time

from numpy.random.mtrand import seed

###  START CODE HERE  ###
# You can tune the hyper-parameter, and define your utility function or class in this block if necessary. 
    
# Particle filter parameter
NP = 1000  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

# Set the random seed to ensure the repeatability.
seed=1
# seed = 2
np.random.seed(seed)

#  Estimation parameter of PF, you may use them in the PF algorithm. You can use the recommended values as follows.
Q = np.diag([0.15]) ** 2  # range error
R = np.diag([0.1, np.deg2rad(10)]) ** 2  # input error

###  END CODE HERE  ###

#  Parameter of LiDAR
scanningAngle = 180
pts=5

#  Simulation parameter
Q_sim = np.diag([0.05]) ** 2  # add noise to lidar readings
# R_sim = np.diag([0, 0]) ** 2  # add noise to control command
R_sim = np.diag([0.03, np.deg2rad(3)]) ** 2  # add noise to control command

v=0.5  #linear velocity
# v=0.48 #linear velocity
w=0.25 #angular velocity

DT = 0.1  # time tick [s]
SIM_TIME = 200.0  # simulation time [s]



class Room(object):
    """
    Generate the map.
    """
    
    def map_range_x(self, start, stop, number, y):
        return [[start + (stop - start) * i / number, y] for i in range(number + 1)]

    def map_range_y(self, start, stop, number, x):
        return [[x, start + (stop - start) * i / number] for i in range(number + 1)]

    def map_square(self, top_left, bottom_right, points):
        tl_x, tl_y = top_left
        br_x, br_y = bottom_right
        res  = self.map_range_y(tl_y, br_y, points, tl_x)
        res += self.map_range_y(tl_y, br_y, points, br_x)
        res += self.map_range_x(tl_x, br_x, points, tl_y)
        res += self.map_range_x(tl_x, br_x, points, br_y)
        return res

    def make_room(self):
        walls = self.map_square((0.0,0.0), (5.0,5.0), 30)
        table1 = self.map_square((2.0,2.0), (3,3.5), 10)
        table2 = self.map_square((2.0,1.0), (2.5,1.5), 5)
        table3 = self.map_square((3.0,1.0), (3.5,1.5), 5)
        table4 = self.map_square((1.0,2.0), (1.5,2.5), 5)
        table5 = self.map_square((4.0,4.0), (4.5,4.5), 5)
        return walls + table1 + table2 + table3 + table4 + table5


def motion_model(x, u):
    """
    Given the current state and control input, return the state at next time.

    Arguments:
    x -- A 3*1 matrix indicating the state of robots or particles. Data format: [[x] [y] [yaw]]
    u -- A 2*1 matrix indicating the control input. Data format: [[v] [w]] 
    
    Return:
    x -- A 3*1 matrix indicating the state at next time. Data format: [[x] [y] [yaw]]
    """

    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
          
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],])
    x = F.dot(x) + B.dot(u)
    return x

def calc_input(v,w):
    """
    Adding noise to the input control commands.

    Argument:
    v -- linear velocity
    w -- angular velocity
    
    Return:
    ud -- A 2*1 matrix indicating the noisy input.
    """

    v = v + np.random.randn() * R_sim[0, 0] ** 0.5
    w = w + np.random.randn() * R_sim[1, 1] ** 0.5
    ud = np.array([[v, w]]).T
    return ud

def generate_particles(pos):
    """
    Generate NP particles, each particle contains three state quantities and weight.  
    If you set the Nearby_flag as True, then the particles will be sprinkled around the real robot.
    Otherwise, the particles will be scattered uniformly in the map.
    In this file, Nearby_flag is set to False to solve a global localization problem. 

    Arguments:
    pos -- The robot current position.

    Return:
    px -- A 3*NP matrix, each column represents the status of a particle.
    pw -- A 1*NP matrix, each column represents the weight value of correspoding particle.
    """
    Nearby_flag = False

    if Nearby_flag:
        x = pos[0] - 1 + 2 * np.random.random(size=(1,NP))
        y = pos[1] - 1 + 2 * np.random.random(size=(1,NP))
        yaw = 2 * math.pi * np.random.random(size=(1,NP))
    else:
        x = 5 * np.random.random(size=(1,NP))
        y = 5 * np.random.random(size=(1,NP))
        yaw = 2 * math.pi * np.random.random(size=(1,NP))
    
    px = np.zeros((3, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    px[0],px[1],px[2],=x,y,yaw
    return px,pw

def notinroom(x, y):
    if(x<=0 or x>=5 or y<=0 or y>=5):
        return False
    if(1.0<=x<=1.5 and 2<=y<=2.5):
        return False
    if(2<=x<=2.5 and 1<=y<=1.5):
        return False
    if(3<=x<=3.5  and 1<=y<=1.5):
        return False
    if(2<=x<=3 and 2<=y<=3.5):
        return False
    if(4<=x<=4.5 and 4<=y<=4.5):
        return False
    return True

def obstacle(px, angle, step = 0.05):
    # return the shortest distance when meeting obstacles
    x, y, theta, dis = px[0][0], px[1][0], px[2][0], 0
    # heng = x+d*cos(theta)      -90  -45     +0   +45     +90     
    # zong = y+d*sin(theta)      -90  -45     +0   +45     +90
    while(notinroom(x,y)):
        x += step * cos(theta + angle)
        y += step * sin(theta + angle)
    return ((px[0][0]-x)**2 + (px[1][0]-y)**2)**0.5

def pf_localization(px,pw,data,u):
    """
    Localization with Particle filter. In this function, you need to:

    (1) Prediction step: Each particle predicts its new location based on the actuation command given.
    (2) Update step:     Update the weight of each particle. That is, particles consistent with sensor readings will have higher weight.
    (3) Resample step:   Generate a set of new particles, with most of them generated around the previous particles with more weight. 
                         You need to decide when to resample the particles and how to resample the particles.
    
    Argument:
    px -- A 3*NP matrix, each column represents the status of a particle.
    pw -- A 1*NP matrix, each column represents the weight value of correspoding particle.
    data -- A List contains the output of the LiDAR sensor. It is represented by a distance value in counterclockwise order.
    u -- A 2*1 matrix indicating the control input. Data format: [[v] [w]] 

    Return:
    x_est -- A 3*1 matrix, indicating the estimated state after particle filtering.
    px -- A 3*NP matrix. The predicted state of the next time.
    pw -- A 1*NP matrix. The updated weight of each particle.
    """
    t1 = time.time()
    ### START CODE HERE ###
    # print(px)
    '''
    motion_model(x, u): Given the current state and control input, return the state at next time.
    heng = x+d*cos(theta)      -90  -45     +0   +45     +90     
    zong = y+d*sin(theta)      -90  -45     +0   +45     +90
    def obstacle(px, angle, step = 0.05):  2 * math.pi 
    '''
    angle = [- 0.5 * math.pi, - 0.25 * math.pi, 0, 0.25 * math.pi, 0.5 * math.pi]
    # u_error = np.random.normal(loc = u, scale = R, size  = NP)
    u_error = np.random.randn(NP,2)@R
    data_error = np.random.randn(NP*5,1)@Q
    # print(data_error)
    for ip in range(NP):
        #  Prediction step: Predict with random input sampling
        # Give the next step state
        tmp_px = np.array([[px[0][ip]],[px[1][ip]],[px[2][ip]]])
        # print(tmp_px)
        # Q[0] * (-1 + 2 * np.random.random())
        # print(np.array([u_error[ip]]).T)
        tmp_px = motion_model(tmp_px,u + np.array([u_error[ip]]).T)
        # print(tmp_px)
        px[0][ip], px[1][ip], px[2][ip] = tmp_px[0], tmp_px[1], tmp_px[2]
        
        #  Update steps: Calculate importance weight
        dis = [abs(obstacle(tmp_px, angle[i])-data[i] - data_error[ip*5+i])**2 for i in range(5)]
        # print(dis)
        # Add the error
        #  + Q[0] * (-1 + 2 * np.random.random())
        pw[0][ip] = 1/np.sum(dis)
        # if(pw[0][ip]<0.1):
        #     pw[0][ip] = 0.02
        #  + np.random.random() * Q[0]
        # print("\t Not Update:")
        # print(pw)
        # pw[ip] = 1/sum(dis) + Q[0]
    pw = pw / pw.sum()  # normalize
    x_est = px.dot(pw.T)

    # Resample step: Resample the particles.
    px, pw = re_sampling(px, pw)
    # print("\t Update")
    # print(pw)
    ###  END CODE HERE  ###
    print("The time used for each iteration:",time.time()-t1," s")
    return x_est,px, pw
    
def re_sampling(px, pw):
    """
    Robot generates a set of new particles, with most of them generated around the previous particles with more weight.

    Argument:
    px -- The state of all particles befor resampling.
    pw -- The weight of all particles befor resampling.

    Return:
    px -- The state of all particles after resampling.
    pw -- The weight of all particles after resampling.
    """
    ### START CODE HERE ###
    # random.random() generate random numbers in [0,1]
    pre_sum = np.zeros(NP)
    weight = np.zeros(NP)
    pre_sum[0] = pw[0][0]
    for i in range(1,NP):
        pre_sum[i] = pre_sum[i-1] + pw[0][i]
    tmp_px = np.zeros((3, NP))
    # generate 400 particles
    for i in range(NP):
        pbt = np.random.random()
        for j in range(NP):
            if(pbt>pre_sum[j]):
                continue
            weight[j] += 1
            # tmp_px[0][i], tmp_px[1][i], tmp_px[2][i] = px[0][j], px[1][j], px[2][j]
            tmp_px[0][i], tmp_px[1][i], tmp_px[2][i] = px[0][j] + Q[0] * (-1 + 2 * np.random.random()), px[1][j] + Q[0] * (-1 + 2 * np.random.random()), px[2][j]
            break
    # print(tmp_px)
    px = tmp_px
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    # pw = np.array([[weight[i] for i in range(NP)]])
    # print(pw)
    # pw = pw / pw.sum()
    ###  END CODE HERE  ###
    return px, pw

if __name__ == '__main__':
    plt.figure(figsize=(8,8)) 
    # Build the room map points.
    r = Room()
    room = r.make_room()  
    # Initialize the controller of robot DR20.
    controller = DR20API.Controller()
    data = controller.get_lidar(Q_sim)
    pos = controller.get_robot_pos()
    ori = controller.get_robot_ori()
    # Generate particles, and the data format is [x y yaw]',[weight].
    px, pw = generate_particles(pos)
    # Use the weighted average to calculate the estimated state.(You can use other way, such as clustering.)
    x_est = np.reshape(px.dot(pw.T),(3,1))
    # Initialize the data history.
    h_x_est = x_est
    h_x_true=np.array([[pos[0]],[pos[1]],[ori]])
    tic = 0.0

    # Initialize the prediction history.
    h_p_true = np.array([[pos[0]],[pos[1]]])
    # print(h_p_true)
    # cnt = 0

    # Start simulation.
    while SIM_TIME >= tic:
        tic += DT
        u = calc_input(v,w)
        # Move the robot
        controller.move_robot_vw(u[0, 0],u[1, 0])
        # Locate the robot
        x_est, px, pw = pf_localization(px, pw, data, u)

        # store data history.
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_true = np.hstack((h_x_true, np.array([[pos[0]],[pos[1]],[ori]])))

         # Get the LiDAR sensor data and real state.
        data = controller.get_lidar(Q_sim)
        pos = controller.get_robot_pos()
        ori = controller.get_robot_ori()
        ###  START CODE HERE  ###
        # Visualization
        if True:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [controller.stop_simulation() if event.key == 'escape' else None])
            x, y = zip(*room)
            plt.scatter(x, y)
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "-b")
            
            # Draw data points
            angle = [- 0.5 * math.pi, - 0.25 * math.pi, 0, 0.25 * math.pi, 0.5 * math.pi]
            xx, yy = [], []
            for i in range(5):
                x, y = pos[0], pos[1]
                x += data[i] * cos(ori + angle[i])
                y += data[i] * sin(ori + angle[i])
                xx.append(x)
                yy.append(y)
            plt.scatter(xx,yy,color = 'r')

            # Plot the particles
            plt.scatter(px[0,:],px[1,:],color = 'g',s=5)

            # Draw the predict path.
            # plt.scatter(sum(px[0,:])/400.0,sum(px[1,:])/400.0,color = 'orange')
            # cnt += 1
            # if(cnt > 15):
            h_p_true = np.hstack((h_p_true, np.array([[sum(px[0,:])/1000.0],[sum(px[1,:])/1000.0]])))
            # print(np.array([[sum(px[0,:])/1000.0],[sum(px[1,:])/1000.0]]))
            plt.plot(np.array(h_p_true[0, :]).flatten(),
                    np.array(h_p_true[1, :]).flatten(), "orange")

            plt.axis("equal")
            plt.grid(True)
            plt.xlim(-0.5,5.5)
            plt.ylim(-0.5,5.5)
            plt.pause(0.01)
        ###  END CODE HERE  ###
    controller.stop_simulation()    