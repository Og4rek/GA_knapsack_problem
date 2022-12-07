import os.path

import numpy as np
from utils import dataReader, arreq_in_list
import random
from math import ceil
from statistics import mean
import matplotlib.pyplot as plt


class itemGenerator:
    @staticmethod
    def generate_items(N):
        weights = list(np.random.randint(1, 11, N))
        profits = list(np.random.randint(1, 11, N))

        return weights, profits


class GAKnapsackSolver:
    def __init__(self, weights, profits, capacity, name, optimal_selection=[]) -> None:
        self.weights = np.array(weights)
        self.profits = np.array(profits)
        self.knapsack_capacity = capacity
        self.optimal_selection = optimal_selection
        self.mutation_rate = 0.3
        self.reproduction_rate = 0.7
        self.crossover_rate = 0.4
        self.name = name
        self.specimen_weight = None
        self.weight_max = None

    def compute_fitness(self):
        wagi = []
        self.fitness = np.empty(len(self.population))
        for i, specimen in enumerate(self.population):
            specimen_fitness = np.dot(specimen, self.profits)
            specimen_weight = np.dot(specimen, self.weights)
            if specimen_weight > self.knapsack_capacity:
                specimen_fitness = 0
            self.fitness[i] = specimen_fitness
            wagi.append(specimen_weight)
        self.weight_mean = np.average(wagi)
        self.weight_max = wagi[np.argmax(self.fitness)]

    def create_initial_population(self, population_size):
        self.population = list()
        while len(self.population) != population_size and 2**len(self.weights) != len(self.population):
            individiual_bits = np.random.randint(2, size=len(self.weights))
            if not arreq_in_list(individiual_bits, self.population):
                self.population.append(individiual_bits)

    # not sure if this method is the correct way to do this
    def tournament_selection_1(self):
        self.compute_fitness()
        temp = list(zip(self.population, self.fitness))
        random.shuffle(temp)
        # print(temp)
        self.population, self.fitness = zip(*temp)
        self.population, self.fitness = list(self.population), list(self.fitness)
        parents = []
        #        for i in range(2):
        temp_population = []
        for j in range(ceil(len(self.population) * 0.5)):
            temp_population.append(random.choice(temp))
        temp_population.sort(key=lambda x: x[1], reverse=True)
        # print(temp_population)
        # print(temp)
        parents.append(temp_population[0][0].astype(int))
        parents.append(temp_population[1][0].astype(int))
        return parents

    def tournament_selection(self):
        parents = []
        while len(parents) != len(self.population):
            temp = list(zip(self.population, self.fitness))
            random.shuffle(temp)
            # print(temp)
            self.population, self.fitness = zip(*temp)
            self.population, self.fitness = list(self.population), list(self.fitness)
            turnament_population = self.population[0:int(len(self.population) * 0.6)]
            turnament_population_fitness = self.fitness[0:int(len(self.fitness) * 0.6)]
            parents.append(turnament_population[np.argmax(turnament_population_fitness)].astype(int))
        return parents

    def one_point_crossover(self, parents):
        length = len(self.weights)
        child1 = np.concatenate((parents[0][:length // 2], parents[1][length // 2:]))
        child2 = np.concatenate((parents[0][length // 2:], parents[1][:length // 2]))
        return [child1, child2]

    def two_point_crossover(self, parents):
        length = len(self.weights)
        p1 = np.random.randint(1, length - 2)
        p2 = np.random.randint(p1+1, length - 1)
        child1 = np.concatenate(
            (parents[0][:p1], parents[1][p1:p2], parents[0][p2:]))
        child2 = np.concatenate(
            (parents[1][:p1], parents[0][p1:p2], parents[1][p2:]))
        return [child1, child2]

    def mutate(self, specimens):
        id_specimens = np.random.randint(0, 2)
        gene = np.random.randint(0, len(specimens[id_specimens]))
        specimens[id_specimens][gene] = np.abs(specimens[id_specimens][gene] - 1)
        for id, specimen in enumerate(specimens):
            for i in range(len(specimen)):
                if random.random() < self.mutation_rate:
                    specimen[i] = np.abs(specimen[i] - 1)
            specimens[id] = specimen
        return specimens

    def create_generation(self):
        next_gen = list()
        children = list()
        random.shuffle(self.tournament_selection())
        parents_list = np.array(self.tournament_selection()).reshape(
            (int(len(self.population) / 2), 2, len(self.weights)))
        for parents in parents_list:
            children = parents
            if random.random() < self.reproduction_rate:
                children = parents
            else:
                if random.random() < self.crossover_rate:
                    # children = self.one_point_crossover(parents)
                    children = self.two_point_crossover(parents)

                if random.random() < self.mutation_rate:
                    children = self.mutate(children)

            next_gen.append(children[0])
            next_gen.append(children[1])
        return next_gen

    def solve(self):
        self.create_initial_population(100)
        average_fitness = []
        max_fitness = []
        generations = []
        weights_mean = []
        weights_max = []
        for generation in range(10000):
            # print(f"...{generation}...")
            self.compute_fitness()
            if 'p' in self.name and arreq_in_list(optimal_selection, self.population):
                # print("Optimal solution reached!")
                max_fitness.append(np.max(self.fitness))
                # print(np.max(self.fitness))
                weights_mean.append(self.weight_mean)
                weights_max.append(self.weight_max)
                average_fitness.append(np.mean(self.fitness))
                generations.append(generation)
                break
            max_fitness.append(np.max(self.fitness))
            average_fitness.append(np.mean(self.fitness))
            generations.append(generation)
            weights_mean.append(self.weight_mean)
            weights_max.append(self.weight_max)
            self.population = self.create_generation()

        self.compute_fitness()
        best_speciman = self.population[np.argmax(self.fitness)]
        print(f"Best specimen: {self.population[np.argmax(self.fitness)]}, Value: {np.max(self.fitness)}")

        plt.plot(generations, average_fitness)
        plt.plot(generations, max_fitness)
        if 'p' in self.name:
            plt.plot(generations[-1], max_fitness[-1], 'r.')
        plt.legend(["Average fitness value", "Maximum fitness value", "Maximum possible fitness"])
        plt.title(f"Fitness coefficent throughout {len(generations)} generations of population")
        plt.xlabel("generation")
        plt.savefig(os.path.join("results", self.name))
        plt.clf()

        plt.plot(generations, weights_mean)
        plt.plot(generations, weights_max)
        plt.plot(generations, np.ones(len(generations)) * self.knapsack_capacity, '--', color='r')
        plt.legend(["Average weights value", "Weights value of maximum fitness value", "Maximum possible weight"])
        plt.title(f"Weights coefficent throughout {len(generations)} generations of population")
        plt.xlabel("generation")
        plt.savefig(os.path.join("results", self.name+"_weights"))
        plt.clf()

        # plt.show()

        return best_speciman


if __name__ == '__main__':
    # capacity_gen = [i*50 for i in range(4, 6)]
    capacity_gen_1 = [200, 250]
    capacity_gen_2 = [600, 800]
    capacity_gen_3 = [1200, 1600]
    weights_gen_1, profits_gen_1 = itemGenerator.generate_items(100)
    weights_gen_2, profits_gen_2 = itemGenerator.generate_items(250)
    weights_gen_3, profits_gen_3 = itemGenerator.generate_items(500)

    version = f'v10000'

    for i, capacity in enumerate(capacity_gen_1):
        genetic_alghoritm_1 = GAKnapsackSolver(weights_gen_1, profits_gen_1, capacity_gen_1[i], f'{version}_g_I100_C{capacity_gen_1[i]}')
        genetic_alghoritm_2 = GAKnapsackSolver(weights_gen_2, profits_gen_2, capacity_gen_2[i], f'{version}_g_I250_C{capacity_gen_2[i]}')
        genetic_alghoritm_3 = GAKnapsackSolver(weights_gen_3, profits_gen_3, capacity_gen_3[i], f'{version}_g_I500_C{capacity_gen_3[i]}')
        best_1 = genetic_alghoritm_1.solve()
        best_2 = genetic_alghoritm_2.solve()
        best_3 = genetic_alghoritm_3.solve()

    filenames = [f'p{i:0>2}' for i in range(1, 9)]
    for i, file in enumerate(filenames):
        weights, profits, capacity, optimal_selection = dataReader.read_data(file)
        genetic_alghoritm = GAKnapsackSolver(weights, profits, capacity, version+'_'+file)
        best = genetic_alghoritm.solve()
