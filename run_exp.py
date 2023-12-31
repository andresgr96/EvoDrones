import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import EA
from EA import run_sim

n_runs = 2
fitnesses = np.zeros(n_runs)

print("-----------------------Beginning Experiment-----------------------")
for i in range(n_runs):
    curr_run = i+1
    print(f"Current Run: {curr_run}")
    fitness = run_sim()
    fitnesses[i] = fitness
    print(f"Run Fitness: {fitness}")

max_fit = np.max(fitnesses)
print("-----------------------Experiment Ended-----------------------")
print(f"Best Fitness: {max_fit}")


