# Genetic Algorithm for the Knapsack Problem

This repository contains the implementation of a genetic algorithm to solve the knapsack problem. The knapsack problem is a popular optimization problem that involves selecting items with given weights and values to maximize the total value without exceeding the weight capacity of the knapsack.

## Introduction

The knapsack problem can be described using a binary vector $X=[x_1,x_2,...,x_N]$, where $x_i=1$ indicates the selection of an item, and $x_i=0$ indicates the exclusion of an item. $N$ represents the number of items. Each item has a specific value $v_i$ and weight $w_i$. The objective is to maximize the total value while ensuring that the total weight does not exceed the knapsack capacity $C$. The constraint and objective function are as follows:

### Constraint

\$$\sum_{i=0}^{N} x_i w_i \leq C\$\$

### Objective Function

$$ f(x,v,w)= \sum_{i=0}^{N}x_iv_i, \qquad \text{if }\sum_{i=0}^{N}x_iw_i \leq C $$

$$ f(x,v,w) = 0, \qquad \text{otherwise} $$

## Solution Description

The process involves the following steps:
1. **Initialization**: Generate an initial population of individuals, where each individual is represented by a genotype $\textbf{V} = [v_1, v_2, ..., v_N]$.
2. **Fitness Calculation**: Calculate the fitness of each individual based on the objective function and store the best individual from the population.
3. **Selection**: Use a tournament selection method to select parents from the current population. Select 60% of individuals randomly and add the best-fitted individual to the parent pool, repeating this $N-1$ times.
4. **Genetic Operations**: Perform crossover and mutation on the parent pool to generate a new subpopulation. Crossover occurs with a probability of 70% using a two-point arithmetic method, while mutation occurs with a probability of 30%, with a 9% chance of mutating other genes.
5. **Iteration**: Repeat the cycle for a specified number of generations or until convergence.

## Results

The algorithm was evaluated for various item quantities $I$ and capacities $C$:
- 100 items
- 250 items
- 500 items

The algorithm generated 1000 generations with a population of 100 individuals for each test case. The results showed a continuous increase in fitness, indicating the algorithm's effectiveness. The detailed results and graphs can be found in the `results` folder.

## Evaluation

The algorithm was further verified using test datasets PO1 and PO7, available on the https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html. The algorithm successfully reached the optimal solution for PO1 in 62 generations and for PO7 in 399 generations.

## Conclusion

The genetic algorithm effectively solves the knapsack problem, as demonstrated by the increasing fitness values and successful optimization on test datasets. The source code for the project is available in this repository.

## Repository Structure

- `data/`: Test datasets.
- `results/`: Evaluation results and graphs.
- `README.md`: This readme file.
- `main.py`: Main script for running the genetic algorithm.
- `utils.py`: Utility functions used in the project.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Og4rek/GA_knapsack_problem.git
```

2. Navigate to the project directory:
```bash
cd GA_knapsack_problem
```

3. Execute the genetic algorithm script:
```bash
python main.py
```

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please open an issue on GitHub or contact the project maintainer.

---
