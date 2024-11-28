import numpy as np
import random
from copy import deepcopy
from models.simulateGame import simulate_game
from models.mlp import MLP


class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.05, crossover_rate=0.7, elitism_count=2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = [MLP() for _ in range(population_size)]

    def evolve_population(self, difficulty):
        new_population = []
        fitness_scores = []
        for nn in self.population:
            score = self.fitness(nn, difficulty)
            fitness_scores.append((nn, score))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        elites = [deepcopy(nn) for nn, _ in fitness_scores[:self.elitism_count]]
        new_population.extend(elites)
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness_scores)
            parent2 = self.select_parent(fitness_scores)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def fitness(self, neural_network, difficulty, verbose=False):
        wins = 0.0
        games = 20
        penalties = 0
        valid_games = 0

        for game_number in range(1, games + 1):
            result = simulate_game(neural_network, difficulty,verbose=verbose, game_number=game_number)
            if result == -0.5:
                penalties += 1
                continue
            valid_games += 1
            if result == 1:
                wins += 1.0
            elif result == 0.5:
                wins += 0.5

        if valid_games == 0:
            return 0
        fitness_value = (wins / valid_games) - (penalties / games * 0.2)
        return max(0, fitness_value)

    def select_parent(self, fitness_scores):
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        elitism = deepcopy(fitness_scores[0][0])
        tournament_size = 5
        tournament = random.sample(fitness_scores, tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)

        if random.random() < 0.3:
            tournament_winner = deepcopy(random.choice(tournament)[0])
        else:
            tournament_winner = deepcopy(tournament[0][0])

        return elitism if random.random() < 0.5 else tournament_winner

    def crossover(self, parent1, parent2):
        child = deepcopy(parent1)
        for attribute in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            parent1_attribute = getattr(parent1, attribute)
            parent2_attribute = getattr(parent2, attribute)
            mask = np.random.rand(*parent1_attribute.shape) < 0.5
            child_attribute = np.where(mask, parent1_attribute, parent2_attribute)
            setattr(child, attribute, child_attribute)
        return child

    def mutate(self, neural_network):
        for attribute in ['weights_input_hidden', 'weights_hidden_output', 'bias_hidden', 'bias_output']:
            tensor = getattr(neural_network, attribute)
            mutation_mask = np.random.rand(*tensor.shape) < self.mutation_rate
            random_values = np.random.uniform(-1.0, 1.0, size=tensor.shape)
            tensor[mutation_mask] += random_values[mutation_mask]