import os
from datetime import datetime

import neat
import pickle
from gym_pybullet_drones.EvoDrones.utils.simulation import run_sim


def eval_genomes(genomes, config):
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        fit = run_sim(genome, config)
        print(i, fit)


def run_neat(config, results_dir):

    # Create a folder with the current date to have better organized results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(results_dir, current_time)
    os.makedirs(experiment_dir)

    
    
    # Start the population and reporters
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_20-59-17/neat-checkpoint38')
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_21-35-56/neat-checkpoint2')
    
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_22-17-51/neat-checkpoint99')
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_22-29-15/neat-checkpoint170')
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_22-44-37/neat-checkpoint240')
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_23-05-10/neat-checkpoint301')
    # p = neat.Checkpointer.restore_checkpoint('results/2024-01-19_23-30-28/neat-checkpoint388')
    p = neat.Checkpointer.restore_checkpoint('results/2024-01-20_00-09-43/neat-checkpoint487')
    
    
    
    
    
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(experiment_dir, 'neat-checkpoint')))

    # Run NEAT and save the best solution to the results dir
    winner = p.run(eval_genomes, 100)
    with open(os.path.join(experiment_dir, "best.pickle"), "wb") as f:
        pickle.dump(winner, f)


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
