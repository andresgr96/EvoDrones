import random
import numpy as np
from simulate import run_sim


# Dummy EA to prepare experiment manager
def evolve_population(generation, population: np.array):
    pop_with_fitness = np.empty((population.shape[0], 2), dtype=object)

    for i, individual in enumerate(population):

        # Get fitness of the individual and save into arr
        fitness = run_sim(individual)
        pop_with_fitness[i] = [np.array(individual), fitness]

    evolved = population * 0.99

    return evolved, pop_with_fitness

