import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
from simulate import run_sim
from evolution import evolve_population
from utils.logger import log_gen

n_gens = 2
population = np.random.uniform(2000, 3000, size=(2, 5))

print("-----------------------Beginning Experiment-----------------------")
for i in range(n_gens):
    curr_gen = i+1
    print(f"Current Generation: {curr_gen}")
    population, pop_with_fits = evolve_population(i, population)

    if curr_gen % 1 == 0:
        log_gen(curr_gen, pop_with_fits)

print("-----------------------Experiment Ended-----------------------")



