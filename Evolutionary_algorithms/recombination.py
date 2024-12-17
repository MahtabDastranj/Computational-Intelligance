import numpy as np
import pandas as pd

generations = 10
gemstones = {'a': [1, 70], 'b': [4, 50], 'c': [6, 30], 'd': [8, 20], 'e': [9, 10], 'f': [7, 40], 'g': [5, 60],
             'h': [3, 80], 'i': [2, 90], 'j': [1, 75], 'k': [1, 25]}
num = len(gemstones)
population = 9
max_weight = 25
crossover_point = 5  # One-point crossover between 5th and 6th gene
mutation_rate = 1  # 1 per offspring

# Convert gemstones to a DataFrame for easy access
gemstone_df = pd.DataFrame(gemstones).T
gemstone_df.columns = ['weight', 'worth']


def calculate_fitness(chromosome):
    total_weight = np.sum(chromosome * gemstone_df['weight'].values)
    total_worth = np.sum(chromosome * gemstone_df['worth'].values)
    return total_worth if total_weight <= max_weight else 0


# Initialize Population Randomly (0s and 1s)
def initialize_population():
    return np.random.randint(2, size=(population, num))


# Select the Best 3 Chromosomes
def select_best(population, fitness):
    sorted_indices = np.argsort(fitness)[::-1]  # Sort fitness in descending order
    return population[sorted_indices[:3]], fitness[sorted_indices[:3]]


# One-Point Crossover
def crossover(parents):
    offspring = []
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            # Perform crossover
            child1 = np.hstack((parents[i][:crossover_point], parents[j][crossover_point:]))
            child2 = np.hstack((parents[j][:crossover_point], parents[i][crossover_point:]))
            offspring.extend([child1, child2])
    return np.array(offspring)


# Mutation: Flip a random gene in each offspring
def mutate(offspring):
    for child in offspring:
        mutation_point = np.random.randint(num)  # Random gene to mutate
        child[mutation_point] = 1 - child[mutation_point]  # Flip the gene (0 -> 1 or 1 -> 0)
    return offspring


# Genetic Algorithm Execution
def genetic_algorithm():
    # Initialize population
    population = initialize_population()
    print("Initial Population:\n", population)

    for generation in range(generations):
        # Step 1: Calculate fitness
        fitness = np.array([calculate_fitness(chromosome) for chromosome in population])
        print(f"\nGeneration {generation + 1} - Fitness: {fitness}")

        # Step 2: Select the best 3 chromosomes
        best_parents, best_fitness = select_best(population, fitness)
        print("Best 3 Parents:\n", best_parents)

        # Step 3: Crossover to generate offspring
        offspring = crossover(best_parents)

        # Step 4: Mutation on offspring
        offspring = mutate(offspring)

        # Step 5: Create new population (3 parents + 6 offspring)
        population = np.vstack((best_parents, offspring))

    # Final Step: Calculate final fitness and select the best solution
    final_fitness = np.array([calculate_fitness(chromosome) for chromosome in population])
    best_index = np.argmax(final_fitness)
    best_chromosome = population[best_index]
    chosen_gems = gemstone_df[best_chromosome == 1]
    total_weight = chosen_gems['weight'].sum()
    total_worth = chosen_gems['worth'].sum()

    print("\nFinal Best Chromosome:", best_chromosome)
    print("Chosen Gemstones:\n", chosen_gems)
    print(f"Total Weight: {total_weight}")
    print(f"Total Worth (Fitness): {total_worth}")


# Run the Genetic Algorithm
genetic_algorithm()
