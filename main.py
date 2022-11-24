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
        self.population, self.fitness = zip(*temp)
        self.population, self.fitness = list(self.population), list(self.fitness)
        parents = []
        for _ in range(len(self.population)):
            for i in range(2):
                temp_population = []
                for j in range(ceil(self.population*0.4)):
                    temp_population.append(random.choice(temp))
                temp_population.sort(key=lambda x: x[1], reverse=False)
                parents.append(temp_population[0].ind.copy())
        return parents


        




        
        

if __name__ == '__main__':
    weights, profits, capacity, optimal_selection = dataReader.read_data("p01")
    genetic_alghoritm = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    genetic_alghoritm.create_initial_population(5)
    genetic_alghoritm.compute_fitness()
    genetic_alghoritm.tournament_selection(5)
    parents = genetic_alghoritm.tournament_selection()
    print(parents)

