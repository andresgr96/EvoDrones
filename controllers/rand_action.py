import random
import numpy as np


# Max RPM: 21702.64377525105
def build_action(num_drones):
    # Four rotors
    action_space = 4
    action = np.zeros((num_drones,action_space))

    # Loop trough each drones actions and create a random value
    for i in range(num_drones):
        for j in range(action_space):
            action[i, j] = random.uniform(14500, 14600)
    # print(action)
    return action

def action_decision(action):
    default = 10000
    if action == 0:
        return np.array([[default-50, default, default, default-50]]) # forward
    elif action == 1:
        return np.array([[default-50, default-50, default, default]]) # right
    elif action == 2:
        return np.array([[default, default-50, default-50, default]]) # backward
    elif action == 3:
        return np.array([[default, default, default-50, default-50]]) # left
    elif action == 4:
        return np.array([[default, default, default, default]]) # up
    
    return np.array([[0, 0, 0, 0]])


def build_action_forward(num_drones):
    side = True       # Fly sideways or upwards (both positively)

    deduction = 50
    default = 15000
    r = np.array([[default-50, default-50, default, default]])
    l = np.array([[default, default, default-50, default-50]])
    f = np.array([[default-50, default, default, default-50]])
    b = np.array([[default, default-50, default-50, default]])
    h = np.array([[default, default, default, default]])
    if side:
        # right : action = np.array([[14950.01, 14950.01, 15000.01, 15000.01]])
        # forward : action = np.array([[14950.01, 15000.01, 15000.01, 14950.01]])
        # backward : action = np.array([[15000.01, 14950.01, 14950.01, 15000.01]])
        # left : action = np.array([[15000.01, 15000.01, 14950.01, 14950.01]])
        
        action = f
    else:
        action = np.array([[15000.01, 15000.01, 14950.01, 14950.01]])
    # print(action)

    return action

def to_hover():
    default = 15000
    h = np.array([[default, default, default, default]])
    return h