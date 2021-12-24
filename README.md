# AI3603_HW

These are homework codes for AI3603 in SJTU. The teacher is Yue Gao. We hope the codes are beneficial for you.

The last update time: 2021/12/24 22:20(Actually the third homework is posted very late.)

2021/12/24 : Released after the assignments are due.

### Homework 1

In this homework, you will develop a path planning framework for a service robot in the unknown environment using A* algorithm. Suppose an autonomous robot DR20 exploring in an unknown scene. The global map is unavailable for the robot at the beginning of exploration, but a laser scanner mounted on robot’s body is utilized to scan the obstacles around the robot. The robot can gradually build the map as it explores, until it reaches the goal.

### Homework 2

There are 2 tasks in this homework. First: In this assignment, you will implement Reinforcement Learning agents to find a safe path to the goal in a grid-shaped maze. The agent will learn by trail and error from interactions with the environment and finally acquire a policy to get as high as possible scores in the game. Second: In this part, you are asked to implement intelligent agents to play the Sokoban game utilizing Sarsa, Qlearning, and dyna-Q algorithms. You will have a deeper understanding on the model-based RL algorithms and the explore-exploit dilemma.

### Homework 3

There is 1 main task in this homework. This assignment is mainly about how to implement the particle filters based on Monto Carlo Localization(MCL) Algorithm. The agent will go arount in a circle and you should know that particle filter is continuously iterated to improve the localization estimate and update localization after the agent moves. This MCL method is mainly about **Prediction, Update and Resample**. You will have a deeper understanding on the MCL algorithm:).

### Environment

These days we found that the environment is very important when we want to use the codes in github. Because without the direct environment we cannot make this work easily and even lead to fail. We are almost hopeless when searching codes in github.

Visual Studio Code: Version 1.60.

Windows environment: Python 3.6. Numpy 1.19.5. Anaconda 3.6 is supported if possible.

The scene model file is not supported on the macOS version of CoppeliaSim. If your device runs on macOS, this makes us sad and we cannot solve it.

Any other problems may be solved by searching on the Internet:).

## Contributor

[AnkorTn](https://github.com/AnkorTn/)

[ZHITENGLI](https://github.com/ZHITENGLI)
