import os
import time
import argparse
from datetime import datetime
import numpy as np
from evolution import evolve_population
from utils.logger import log_gen

import neat
import pickle
from simulate import run_sim

def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        run_sim(genome, config)
        
    
def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    winner = p.run(eval_genomes, 5)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    print(local_dir)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    run_neat(config)




