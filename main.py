import numpy as np
from utils import dataReader, arreq_in_list
import random
from math import ceil
from statistics import mean
import matplotlib.pyplot as plt

class GAKnapsackSolver:
    def __init__(self, weights, profits, capacity, optimal_selection) -> None:
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.knapsack_capacity = capacity 
        self.optimal_selection = optimal_selection
        self.mutation_rate = 0.3
        self.reproduction_rate = 0.7
        self.crossover_rate = 0.4

    def compute_fitness(self):
        self.fitness = np.empty(len(self.population))
        for i,specimen in enumerate(self.population):
            specimen_fitness = np.dot(specimen, self.profits)
            specimen_weight = np.dot(specimen, self.weights)
            if specimen_weight > self.knapsack_capacity:
                specimen_fitness = 0
            self.fitness[i] = specimen_fitness
    
    def get_fitness(self):
        return self.fitness


    def create_initial_population(self, population_size):
        self.population = list()
        while len(self.population) != population_size:
            individiual_bits = np.random.randint(2, size=len(self.weights))
            if not arreq_in_list(individiual_bits, self.population):
                self.population.append(individiual_bits)
    
    #not sure if this method is the correct way to do this
    def tournament_selection_1(self):
        self.compute_fitness()
        temp = list(zip(self.population, self.fitness))
        random.shuffle(temp)
        #print(temp)
        self.population, self.fitness = zip(*temp)
        self.population, self.fitness = list(self.population), list(self.fitness)
        parents = []
        for i in range(2):
            temp_population = []
            for j in range(ceil(len(self.population)*0.7)):
                temp_population.append(random.choice(temp))
            temp_population.sort(key=lambda x: x[1], reverse=True)
            #print(temp_population)
           # print(temp)
            parents.append(temp_population[0][0].astype(int))
        return parents

    def tournament_selection(self):
        parents = []
        while len(parents) != len(self.population):
            self.compute_fitness()
            temp = list(zip(self.population, self.fitness))
            random.shuffle(temp)
            #print(temp)
            self.population, self.fitness = zip(*temp)
            self.population, self.fitness = list(self.population), list(self.fitness)
            turnament_population = self.population[0:int(len(self.population) * 0.6)]
            turnament_population_fitness = self.fitness[0:int(len(self.fitness) * 0.6)]
            parents.append(turnament_population[np.argmax(turnament_population_fitness)])
        return parents

    def one_point_crossover(self, parents):
        length = len(self.weights)
        child1 = np.concatenate((parents[0][:length//2], parents[1][length//2:]))
        child2 = np.concatenate((parents[0][length//2:], parents[1][:length//2]))
        return [child1, child2]

    def mutate(self, specimens):
        for id, specimen in enumerate(specimens):
            for i in range(len(specimen)):
                if random.random() < self.mutation_rate:
                    if specimen[i] == 0:
                        specimen[i] = 1
                    else:
                        specimen[i] = 0
            specimens[id] = specimen
            return specimens

    def create_generation(self):
        next_gen = []
        children = list()
        parents_list = np.array(self.tournament_selection()).reshape((25, 2, 10))
        for parents in parents_list:
            children = parents
            if random.random() < self.reproduction_rate:
                children = parents
            else:
                if random.random() < self.crossover_rate:
                    children = self.one_point_crossover(parents)
                
                if random.random() < self.mutation_rate:
                    children = self.mutate(children)

            next_gen.append(children[0])
            next_gen.append(children[1])
        return next_gen
    
    def solve(self):
        self.create_initial_population(50)
        self.compute_fitness()
        average_fitness = []
        generations = []
        for generation in range(500):
            average_fitness.append(mean(self.get_fitness()))
            self.population = self.create_generation()
            generations.append(generation)
            # print(generation)

        self.compute_fitness()
        best_speciman = self.population[np.argmax(self.fitness)]
        print(f"Best specimen: {self.population[np.argmax(self.fitness)]}, Value: {np.max(self.fitness)}")
        
        plt.plot(generations, average_fitness)
        plt.show()
        
        return best_speciman


if __name__ == '__main__':
    weights, profits, capacity, optimal_selection = dataReader.read_data("p01")
    genetic_alghoritm = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    best = genetic_alghoritm.solve()

