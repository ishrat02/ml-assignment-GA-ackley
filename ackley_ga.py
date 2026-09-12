"""
Genetic Algorithm Assignment
============================
Task: Find the global minimum of the Ackley function (2D) using a
      Genetic Algorithm.

Ackley function (2D):
    f(x1, x2) = -a*exp(-b*sqrt(0.5*(x1^2+x2^2)))
                - exp(0.5*(cos(c*x1) + cos(c*x2)))
                + a + e

Parameters : a = 20, b = 0.2, c = 2*pi
Global min : f(x*) = 0 at x* = (0, 0)
Search space: -5 <= x1, x2 <= 5

GA settings (as specified in the assignment):
    Population size   : 50
    Generations       : 100
    Crossover prob Pc : 0.80
    Mutation  prob Pm : 0.05
    Elitism           : 1 (best individual copied unchanged each gen)
    Encoding          : Value (real) encoding -> chromosome = (x1, x2)
    Selection         : Roulette wheel selection
    Crossover         : One-point crossover (point after gene 1,
                         i.e. x1 and x2 are swapped between parents)
    Mutation          : For each gene, with probability Pm, perturb it
                         with a random value (Gaussian) and clip back
                         into the [-5, 5] boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Ackley function (objective / fitness basis)
# ---------------------------------------------------------------
A, B, C = 20.0, 0.2, 2 * np.pi
LOWER, UPPER = -5.0, 5.0


def ackley(x1, x2):
    term1 = -A * np.exp(-B * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
    term2 = -np.exp(0.5 * (np.cos(C * x1) + np.cos(C * x2)))
    return term1 + term2 + A + np.e


# ---------------------------------------------------------------
# 2. GA settings
# ---------------------------------------------------------------
POP_SIZE = 50
GENERATIONS = 100
PC = 0.80          # crossover probability
PM = 0.05          # mutation probability
ELITISM = 1
MUTATION_STRENGTH = 1.0   # std-dev of Gaussian perturbation


# ---------------------------------------------------------------
# 3. Initialization (value encoding: chromosome = [x1, x2])
# ---------------------------------------------------------------
def init_population(pop_size):
    return np.random.uniform(LOWER, UPPER, size=(pop_size, 2))


# ---------------------------------------------------------------
# 4. Fitness calculation
#    This is a MINIMIZATION problem, but roulette wheel selection
#    needs "higher fitness -> bigger slice of the pie chart".
#    So we convert the objective value into a fitness score where
#    a LOWER Ackley value gives a HIGHER fitness.
# ---------------------------------------------------------------
def evaluate(population):
    f_values = np.array([ackley(ind[0], ind[1]) for ind in population])
    fitness = 1.0 / (1.0 + f_values)   # smaller f(x) -> larger fitness
    return f_values, fitness


# ---------------------------------------------------------------
# 5. Roulette wheel selection
# ---------------------------------------------------------------
def roulette_wheel_selection(population, fitness):
    probs = fitness / fitness.sum()
    cumulative = np.cumsum(probs)
    r = np.random.rand()
    idx = np.searchsorted(cumulative, r)
    idx = min(idx, len(population) - 1)
    return population[idx]


# ---------------------------------------------------------------
# 6. One-point crossover
#    Only one possible cut point for a 2-gene chromosome: after gene 1.
#    Offspring 1 = (parent1.x1, parent2.x2)
#    Offspring 2 = (parent2.x1, parent1.x2)
# ---------------------------------------------------------------
def crossover(parent1, parent2):
    if np.random.rand() < PC:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    return child1, child2


# ---------------------------------------------------------------
# 7. Mutation
#    Traverse each gene; with probability Pm, add a random
#    (Gaussian) perturbation, then clip back into the boundary.
# ---------------------------------------------------------------
def mutate(individual):
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < PM:
            child[i] += np.random.normal(0, MUTATION_STRENGTH)
            child[i] = np.clip(child[i], LOWER, UPPER)
    return child


# ---------------------------------------------------------------
# 8. Main GA loop
# ---------------------------------------------------------------
def genetic_algorithm(verbose=True):
    population = init_population(POP_SIZE)
    best_x = None
    best_f = np.inf
    history = []

    for gen in range(1, GENERATIONS + 1):
        f_values, fitness = evaluate(population)

        # Track best-so-far
        gen_best_idx = np.argmin(f_values)
        if f_values[gen_best_idx] < best_f:
            best_f = f_values[gen_best_idx]
            best_x = population[gen_best_idx].copy()
        history.append(best_f)

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"Generation {gen:3d} | Best f(x) so far = {best_f:.6f} "
                  f"| x = ({best_x[0]:.4f}, {best_x[1]:.4f})")

        # Elitism: carry the best individual(s) forward untouched
        elite_idx = np.argsort(f_values)[:ELITISM]
        new_population = [population[i].copy() for i in elite_idx]

        # Fill the rest of the new population
        while len(new_population) < POP_SIZE:
            p1 = roulette_wheel_selection(population, fitness)
            p2 = roulette_wheel_selection(population, fitness)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutate(c1), mutate(c2)
            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)

        population = np.array(new_population)

    return best_x, best_f, history


# ---------------------------------------------------------------
# 9. Run + report + plot convergence
# ---------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)  # remove/change for different random runs

    best_x, best_f, history = genetic_algorithm()

    print("\n===== RESULT =====")
    print(f"Best solution : x1 = {best_x[0]:.6f}, x2 = {best_x[1]:.6f}")
    print(f"Ackley f(x)   : {best_f:.6f}")
    print("Known global minimum: f(0, 0) = 0")

    # Convergence plot
    plt.figure(figsize=(7, 5))
    plt.plot(history, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Ackley value found so far")
    plt.title("GA Convergence on 2D Ackley Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ga_ackley_convergence.png", dpi=150)
    print("\nConvergence plot saved as 'ga_ackley_convergence.png'")
