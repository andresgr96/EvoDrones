import numpy as np


def log_gen(gen, pop_with_fits: np.array):

    print(f"-----------------------Stats at Generation {gen}-----------------------")
    max_idx = np.argmax(pop_with_fits[:, 1])
    min_idx = np.argmin(pop_with_fits[:, 1])

    max_fitness = pop_with_fits[max_idx][1]
    min_fitness = pop_with_fits[min_idx][1]
    mean_fitness = np.mean(pop_with_fits[:, 1])

    print(f"Max Fitness: {max_fitness}")
    print(f"Min Fitness: {min_fitness}")
    print(f"Mean Fitness: {mean_fitness}")
