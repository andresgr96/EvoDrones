import os
import time
import argparse
from datetime import datetime
import numpy as np
from evolution import evolve_population
from utils.logger import log_gen
import functools


import neat
import pickle
from simulate import run_sim
from multiprocessing import Pool


# Evaluates a single genome and assigns its fitness
def eval_genome(genome_data):
    global run_sim
    genome_id, genome, config = genome_data

    print(f"Before: {genome.fitness}")
    genome.fitness = run_sim(genome, config)
    print(f"After: {genome.fitness}")


# Evaluates a batch of genomes in parallel, library takes care of creating n cores where n = cpu cores of ur pc
def eval_genomes_parallel(genomes, config):
    with Pool() as pool:
        try:
            # Use map_async to pass a single argument (genome) to eval_genome
            results = pool.map_async(eval_genome, [(genome_id, genome, config) for genome_id, genome in genomes])

            # Wait for all evaluations to complete
            results.wait()
        except Exception as e:
            print(f"Exception in eval_genomes_parallel: {e}")
            # Handle the exception as needed




# Run the entire neat algo
def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # # Initialize fitness for all genomes
    # initialize_fitness(p.population)

    # # Create a list of (genome_id, genome) tuples to pass to eval_genome
    # genomes = [(genome_id, genome) for genome_id, genome in p.population.items()]
    #
    # # Evaluate genomes in parallel
    # eval_genomes_parallel(genomes, config)

    # Use p.population to get the winners and save them as needed
    winner = p.run(eval_genomes_parallel, 5)
    with open("822-45-strict.pickle", "wb") as f:
        pickle.dump(winner, f)


# Main Runner
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    print(local_dir)
    config_path = os.path.join(local_dir, '../assets/config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    run_neat(config)
