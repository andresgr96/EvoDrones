import neat
import pickle
import gzip
import random
from neat.population import Population

def inspect_checkpoint(checkpoint_file):
    # Create a Checkpointer object and restore the checkpoint
    checkpointer = neat.Checkpointer()
    # population = checkpointer.restore_checkpoint(checkpoint_file)
    # print(f"Generation: {checkpoint_file[-3:]}")
    # print(f"Population Size: {len(population.population)}")
    # # print(f"Best Fitness: {population.population.}")
    # # for genome in population:
    # #     print(genome.fitness)
    #
    # if population is None:
    #     print("Error: Could not restore checkpoint.")
    #     return

    with gzip.open(checkpoint_file) as f:
        generation, config, population, species_set, rndstate = pickle.load(f)
        restored_population = Population(config, (population, species_set, generation))
        print(f"Generation: {generation}")
        print(f"Population Size: {len(population)}")

        # Filter out genomes with None fitness
        valid_genomes = {k: v for k, v in restored_population.population.items() if v.fitness is not None}

        if valid_genomes:
            best_genome = max(valid_genomes.items(), key=lambda x: x[1].fitness)
            print(f"Best genome ID: {best_genome[0]}")
            print(f"Best genome fitness: {best_genome[1].fitness}")
        else:
            print("No valid genomes with fitness values found.")


# Specify the path to the checkpoint file
checkpoint_file = "results/2024-02-18_17-52-25/neat-checkpoint110"

# Call the function to inspect the checkpoint
inspect_checkpoint(checkpoint_file)
