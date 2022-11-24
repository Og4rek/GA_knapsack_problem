import numpy as np
from utils import dataReader, arreq_in_list
import random
from math import ceil

class GAKnapsackSolver:
    def __init__(self, weights, profits, capacity, optimal_selection) -> None:
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.knapsack_capacity = capacity 
        self.optimal_selection = optimal_selection

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
            individiual_bits = np.random.randint(2, size=len(self.weights))
            if not arreq_in_list(individiual_bits, self.population):
                self.population.append(individiual_bits)
    
    #not sure if this method is the correct way to do this
    def tournament_selection(self):
        temp = list(zip(self.population, self.fitness))
        random.shuffle(temp)
        #print(temp)
        self.population, self.fitness = zip(*temp)
        self.population, self.fitness = list(self.population), list(self.fitness)
        parents = []
        for i in range(2):
            temp_population = []
            for j in range(ceil(len(self.population)*0.4)):
                temp_population.append(random.choice(temp))
            temp_population.sort(key=lambda x: x[1], reverse=False)
            parents.append(temp_population[0][0].astype(int))
        return parents

    def one_point_crossover(self, parents):
        length = len(self.weights)
        child1 = np.concatenate((parents[0][:length//2],parents[1][length//2:]))
        child2 = np.concatenate((parents[0][length//2:],parents[1][:length//2]))
        return [child1, child2]

if __name__ == '__main__':
    weights, profits, capacity, optimal_selection = dataReader.read_data("p01")
    genetic_alghoritm = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    genetic_alghoritm.create_initial_population(20)
    genetic_alghoritm.compute_fitness()
    parents = genetic_alghoritm.tournament_selection()
    print(parents)
    print(genetic_alghoritm.one_point_crossover(parents))

