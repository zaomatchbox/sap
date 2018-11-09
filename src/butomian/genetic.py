import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import nps_chat
import numpy as np

import random
import math
from deap import base, creator, tools
from collections import Counter

NUM_BITS = 100
random.seed(7)


def diff(x, y):
    return x - y


def loss(x):
    res = 0
    mid = np.mean(x)
    count = sum([1 if _ > mid else 0 for _ in x])
    left = x[:count]
    right = x[count:]
    for _ in left:
        res += _ if _ > mid else 0
    for _ in right:
        res += 2 * _ if _ < mid else 0
    return res


def create_toolbox(bits=NUM_BITS, weights=(1.0, )):
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, NUM_BITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", loss)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


if __name__ == "__main__":
    toolbox = create_toolbox(bits=NUM_BITS)
    population = toolbox.population(n=1000)
    probab_crossing, probab_mutating = 0.7, 0.3
    num_generations = 100
    print('Evolution process starts')
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit, )
    print('\nEvaluated', len(population), 'individuals')
    for g in range(num_generations):
        print("\n- Generation", g)

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probab_crossing:
                toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:
            if random.random() < probab_mutating:
                toolbox.mutate(mutant)
            del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit, )
        print('Evaluated', len(invalid_ind), 'individuals')
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print('Min =', min(fits), ', Max =', max(fits))
        print('Average =', round(mean, 2), ', Standard deviation =',
              round(std, 2))
        print("\n- Evolution ends")

    best_ind = tools.selBest(population, 1)[0]
    print('\nBest individual:\n', best_ind)

    n = int(len(best_ind) / 2)
    print(best_ind[:n])
    print(best_ind[n:])