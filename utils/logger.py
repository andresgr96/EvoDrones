import numpy as np
import os


def log_gen(exp_name, gen, pop_with_fits: np.array):

    print(f"-----------------------Stats at Generation {gen}-----------------------")
    max_idx = np.argmax(pop_with_fits[:, 1])
    min_idx = np.argmin(pop_with_fits[:, 1])

    best_solution = pop_with_fits[max_idx]
    max_fitness = best_solution[1]
    min_fitness = pop_with_fits[min_idx][1]
    mean_fitness = np.mean(pop_with_fits[:, 1])

    print(f"Max Fitness: {max_fitness}")
    print(f"Min Fitness: {min_fitness}")
    print(f"Mean Fitness: {mean_fitness}")

    results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

    # Create or open the text file for logging
    log_file_path = os.path.join(results_folder, f"{exp_name}_results.txt")
    with open(log_file_path, "a") as log_file:
        # If the file doesn't exist, create it and write a header
        if not os.path.exists(log_file_path):
            log_file.write("Generation,Max_Fitness,Min_Fitness,Mean_Fitness\n")

        # Write the information in the same line
        log_file.write(f"{gen},{max_fitness},{min_fitness},{mean_fitness}, {best_solution}\n")


# exp_name = "Experiment1"
# gen = 1
# pop_with_fits = np.random.rand(10, 2)  # Replace this with your actual data
#
# log_gen(exp_name, gen, pop_with_fits)