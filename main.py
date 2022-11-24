import numpy as np
from utils import Item, dataReader, arreq_in_list

class GAKnapsackSolver:
    def __init__(self, weights, profits, capacity, optimal_selection) -> None:
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.knapsack_capacity = capacity 
        self.optimal_selection = optimal_selection
        self.item_list = []

    def compute_fitness(self):
        self.fitness = np.empty(len(self.population))
        for i,specimen in enumerate(self.population):
            specimen_fitness = np.dot(specimen, self.profits)
            specimen_weight = np.dot(specimen, self.weights)
            if specimen_weight > self.knapsack_capacity:
                specimen_fitness = 0
            self.fitness[i] = specimen_fitness


    def create_initial_population(self, population_size):
        self.population = list()
        while len(self.population) != population_size:
            individiual_bits = np.random.randint(2, size=len(self.item_list))
            if not arreq_in_list(individiual_bits, self.population):
                self.population.append(individiual_bits)

        

if __name__ == '__main__':
    weights, profits, capacity, optimal_selection = dataReader.read_data("p01")

    genetic_alghoritm = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    genetic_alghoritm.create_initial_population(5)
    genetic_alghoritm.compute_fitness()
    print(genetic_alghoritm.fitness)

