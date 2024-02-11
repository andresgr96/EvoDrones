import random
import numpy as np


# Dummy predict function to prepare the experiment manager
def predict(state: np.array, individual: np.array):

    total = np.sum(state)
    rpms = np.random.choice(individual, size=4, replace=False)
    rpms = rpms * total

    return rpms


def build_action(action) -> np.array:
    base_rpm = 14100
    potential_rpm = 600
    action = np.array(action)
    action = action * potential_rpm

    action = action + base_rpm

    return np.array([action])

def build_action_range(action) -> np.array:
    action = np.array(action)
    normalized_action = action #* 100000

    for i in range(len(normalized_action)):
        if normalized_action[i] > 16000:
            normalized_action[i] = 16000  # Clip the value to the maximum allowed RPM
        if normalized_action[i] < 14500:
            normalized_action[i] = 14500  # Clip the value to the maximum allowed RPM

    return np.array([normalized_action])



