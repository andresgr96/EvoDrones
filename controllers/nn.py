import random
import numpy as np


# Dummy predict function to prepare the experiment manager
def predict(state: np.array, individual: np.array):

    total = np.sum(state)
    rpms = np.random.choice(individual, size=4, replace=False)
    rpms = rpms * total

    return rpms


