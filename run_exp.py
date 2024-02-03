import os
from datetime import datetime

import matplotlib.pyplot as plt
import warnings
import numpy as np

import neat
import pickle
import matplotlib.pyplot as plt
import warnings
import numpy as np
from gym_pybullet_drones.EvoDrones.simulators.simulation import run_sim


def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        run_sim(genome, config)
        if i % 10 == 0:
            print(i)

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    # plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    # plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
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
    
def run_neat(config, results_dir):

    # Create a folder with the current date to have better organized results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, current_time)
    os.makedirs(experiment_dir)

    # Start the population and reporters
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-25_17-59-43/neat-checkpoint19')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(experiment_dir, 'neat-checkpoint')))

    # Run NEAT and save the best solution to the results dir
    winner = p.run(eval_genomes, 40)
    with open(os.path.join(experiment_dir, "best.pickle"), "wb") as f:
        pickle.dump(winner, f)
        
    plot_stats(stats, ylog=False, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)

    # Specify the results and config directories
    results_dir = os.path.join(local_dir, "results")
    config_dir = os.path.join(local_dir, "assets")

    # Setup NEAT configs
    config_path = os.path.join(config_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config, results_dir)
