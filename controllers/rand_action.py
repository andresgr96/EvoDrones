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


def build_action_forward(num_drones):
    side = True       # Fly sideways or upwards (both positively)

    if side:
        action = np.array([[14950.01, 15000.01, 15000.01, 14950.01]])
    else:
        action = np.array([[15000.01, 15000.01, 14950.01, 14950.01]])
    # print(action)

    return action

