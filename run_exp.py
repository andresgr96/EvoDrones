import os
from datetime import datetime

import neat
import pickle
import matplotlib.pyplot as plt
import warnings
import numpy as np


def eval_genomes(genomes, config):
    # global controller

    print(controller)
    if controller == "takeoff":
        from gym_pybullet_drones.evo_drones.simulators.simulation_exp_one_rpms_takeoff import run_sim
    elif controller == "follow":
        from gym_pybullet_drones.evo_drones.simulators.simulation_exp_one_rpms_follow import run_sim
    else:
        from gym_pybullet_drones.evo_drones.simulators.simulation_exp_one_rpms_land import run_sim

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        run_sim(genome, config)


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    avg_fitness = np.array(statistics.get_fitness_mean())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.title("Population's average fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def run_neat(config, results_dir, checkpoint=None):

    # Create a folder with the current date to have better organized results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, current_time)
    os.makedirs(experiment_dir)

    # Start the population and reporters
    if checkpoint:
        p = neat.Checkpointer().restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(experiment_dir, 'neat-checkpoint')))

    # Run NEAT and save the best solution to the results dir
    winner = p.run(eval_genomes, 1)

    with open(os.path.join(experiment_dir, f"{controller}_best.pickle"), "wb") as f:
        pickle.dump(winner, f)

    plot_stats(stats, ylog=False, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)

    # Specify the results and config directories
    results_dir = os.path.join(local_dir, "results")
    config_dir = os.path.join(local_dir, "envs/assets")

    # Setup NEAT configs
    config_path = os.path.join(config_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Specify the checkpoint file if resuming from a checkpoint
    # checkpoint_file = os.path.join(results_dir, '2024-03-02_23-27-52', 'neat-checkpoint406')
    checkpoint_file = None

    # Declare which controller to train
    global controller
    controller = "takeoff"

    # Change checkpoint to None if you dont want to resume training.
    run_neat(config, results_dir, checkpoint=checkpoint_file)
