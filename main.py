import numpy as np
from utils import dataReader, arreq_in_list
import random
from math import ceil
from statistics import mean
import matplotlib.pyplot as plt

class itemGenerator:
    @staticmethod
    def generate_items(N):
        weights = list(np.random.randint(1,11,N))
        profits = list(np.random.randint(1,11,N))

        return weights, profits

class GAKnapsackSolver:
    def __init__(self, weights, profits, capacity, optimal_selection = []) -> None:
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
#        for i in range(2):
        temp_population = []
        for j in range(ceil(len(self.population)*0.5)):
            temp_population.append(random.choice(temp))
        temp_population.sort(key=lambda x: x[1], reverse=True)
        #print(temp_population)
        # print(temp)
        parents.append(temp_population[0][0].astype(int))
        parents.append(temp_population[1][0].astype(int))
        return parents

    def tournament_selection(self):
        parents = []
        while len(parents) != len(self.population):
            temp = list(zip(self.population, self.fitness))
            random.shuffle(temp)
            #print(temp)
            self.population, self.fitness = zip(*temp)
            self.population, self.fitness = list(self.population), list(self.fitness)
            turnament_population = self.population[0:int(len(self.population) * 0.6)]
            turnament_population_fitness = self.fitness[0:int(len(self.fitness) * 0.6)]
            parents.append(turnament_population[np.argmax(turnament_population_fitness)].astype(int))
        return parents

    def one_point_crossover(self, parents):
        length = len(self.weights)
        child1 = np.concatenate((parents[0][:length//2], parents[1][length//2:]))
        child2 = np.concatenate((parents[0][length//2:], parents[1][:length//2]))
        return [child1, child2]

    def two_point_crossover(self, parents):
        length = len(self.weights)
        child1 = np.concatenate((parents[0][:length//3],parents[1][length//3:(length//3)*2],parents[0][(length//3)*2:]))
        child2 = np.concatenate((parents[1][:length//3],parents[0][length//3:(length//3)*2],parents[1][(length//3)*2:]))
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
        parents_list = np.array(self.tournament_selection()).reshape((int(len(self.population)/2), 2, len(self.weights)))
        for parents in parents_list:
            children = parents
            if random.random() < self.reproduction_rate:
                children = parents
            else:
                if random.random() < self.crossover_rate:
                    #children = self.one_point_crossover(parents)
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
        for generation in range(100):
            print(f"...{generation}...")
            self.compute_fitness()
            # if arreq_in_list(optimal_selection, self.population):
            #     print("Optimal solution reached!")
            #     max_fitness.append(np.max(self.fitness))
            #     print(np.max(self.fitness))
            #     average_fitness.append(np.mean(self.fitness))
            #     generations.append(generation)
            #     break
            max_fitness.append(np.max(self.fitness))
            average_fitness.append(np.mean(self.fitness))
            generations.append(generation)
            self.population = self.create_generation()
    

        self.compute_fitness()
        best_speciman = self.population[np.argmax(self.fitness)]
        print(f"Best specimen: {self.population[np.argmax(self.fitness)]}, Value: {np.max(self.fitness)}")
        
        plt.plot(generations, average_fitness)
        plt.plot(generations, max_fitness)
        plt.plot(generations[-1], max_fitness[-1], 'r.')
        plt.legend(["Average fitness value", "Maximum fitness value", "Maximum possible fitness"])
        plt.title(f"Fitness coefficent throughout {len(generations)} generations of population")
        plt.xlabel("generation")
        plt.show()
        
        return best_speciman


if __name__ == '__main__':
    capacity_gen = 70
    weights_gen, profits_gen = itemGenerator.generate_items(50)

    weights, profits, capacity, optimal_selection = dataReader.read_data("p02")
    genetic_alghoritm = GAKnapsackSolver(weights_gen, profits_gen, capacity_gen)
    best = genetic_alghoritm.solve()



    # weights, profits, capacity, optimal_selection = dataReader.read_data("p01")
    # genetic_alghoritm_1 = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    # weights, profits, capacity, optimal_selection = dataReader.read_data("p02")
    # genetic_alghoritm_2 = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    # weights, profits, capacity, optimal_selection = dataReader.read_data("p05")
    # genetic_alghoritm_3 = GAKnapsackSolver(weights, profits, capacity, optimal_selection)
    # weights, profits, capacity, optimal_selection = dataReader.read_data("p08")
    # genetic_alghoritm_4 = GAKnapsackSolver(weights, profits, capacity, optimal_selection)

    # best_speciman_1, generations_1, max_fitness_1, average_fitness_1 = genetic_alghoritm_1.solve(50)
    # best_speciman_2, generations_2, max_fitness_2, average_fitness_2 = genetic_alghoritm_2.solve(6)
    # best_speciman_3, generations_3, max_fitness_3, average_fitness_3 = genetic_alghoritm_3.solve(50)
    # best_speciman_4, generations_4, max_fitness_4, average_fitness_4 = genetic_alghoritm_4.solve(50)
    # fig, axs = plt.subplots(2, 2)
    # fig.suptitle("Fitness coefficent throughout generations of population")
    # axs[0, 0].plot(generations_1, average_fitness_1)
    # axs[0, 0].plot(generations_1, max_fitness_1)
    # axs[0, 0].set_title(f'[p01] {len(generations_1)} generations')
    # axs[0, 1].plot(generations_2, average_fitness_2)
    # axs[0, 1].plot(generations_2, max_fitness_2)
    # axs[0, 1].set_title(f'[p02] {len(generations_2)} generations')
    # axs[1, 0].plot(generations_3, average_fitness_3)
    # axs[1, 0].plot(generations_3, max_fitness_3)
    # axs[1, 0].set_title(f'[p05] {len(generations_2)} generations')
    # axs[1, 1].plot(generations_4, average_fitness_4)
    # axs[1, 1].plot(generations_4, max_fitness_4)
    # axs[1, 1].set_title(f'[p05] {len(generations_4)} generations')