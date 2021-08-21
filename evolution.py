import copy
import json
import os

from player import Player
import numpy as np
from config import CONFIG


class Evolution:

    def __init__(self, mode):
        self.mode = mode
        self.generation_number = 0

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def add_gaussian_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)

    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.2

        self.add_gaussian_noise(child.nn.W1, threshold)
        self.add_gaussian_noise(child.nn.W2, threshold)
        self.add_gaussian_noise(child.nn.b1, threshold)
        self.add_gaussian_noise(child.nn.b2, threshold)

    def roulette_wheel(self, players, parent_numbers):
        probabilities = self.calculate_cumulative_probabilities(players)

        results = []
        for random_number in np.random.uniform(low=0, high=1, size=parent_numbers):
            for i, probability in enumerate(probabilities):
                if random_number <= probability:
                    results.append(copy.deepcopy(players[i]))
                    break

        return results

    def calculate_cumulative_probabilities(self, players):
        total_fitness = 0
        for player in players:
            total_fitness += player.fitness
        probabilities = []
        for player in players:
            probabilities.append(player.fitness / total_fitness)
        # turn it to cumulative probability
        for i in range(1, len(players)):
            probabilities[i] += probabilities[i - 1]
        return probabilities

    # def crossover(self, array1, array2, mode):
    #     if array1.shape != array2.shape:
    #         print("ERROR IN CROSSOVER")
    #         exit()
    #
    #     row_size, column_size = array1.shape
    #     # section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size
    #     # if mode == 0:
    #     #     array1[:section_1, :] = array2[:section_1, :]
    #     #     array1[section_2:, :] = array2[section_2:, :]
    #     # elif mode == 1:
    #     #     array1[section_1:section_2, :] = array2[section_1:section_2, :]
    #
    #     half_of_size = int(row_size / 2)
    #     if mode == 0:
    #         array1[:half_of_size, :] = array2[:half_of_size, :]
    #     elif mode == 1:
    #         array1[half_of_size:, :] = array2[half_of_size:, :]

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        row_size, column_size = child1_array.shape
        section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:section_1, :] = parent1_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent1_array[section_2:, :]

            child2_array[:section_1, :] = parent2_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent2_array[section_2:, :]
        else:
            child1_array[:section_1, :] = parent2_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent2_array[section_2:, :]

            child2_array[:section_1, :] = parent1_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent1_array[section_2:, :]

    def reproduction(self, parent1, parent2):
        child1 = Player(self.mode)
        child2 = Player(self.mode)

        self.crossover(child1.nn.W1, child2.nn.W1, parent1.nn.W1, parent2.nn.W1)
        self.crossover(child1.nn.W2, child2.nn.W2, parent1.nn.W2, parent2.nn.W2)
        self.crossover(child1.nn.b1, child2.nn.b1, parent1.nn.b1, parent2.nn.b1)
        self.crossover(child1.nn.b2, child2.nn.b2, parent1.nn.b2, parent2.nn.b2)

        self.mutate(child1)
        self.mutate(child2)
        return child1, child2

    def q_tournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            # num_players example: 150
            # prev_players: an array of `Player` objects
            method = "Q tournament"
            children = []
            parents = []

            if method == 'roulette wheel':
                parents = self.roulette_wheel(prev_players, num_players)
            elif method == "Q tournament":
                for _ in range(num_players):
                    parents.append(self.q_tournament(prev_players, q=3))

            for i in range(0, len(parents), 2):
                child1, child2 = self.reproduction(parents[i], parents[i + 1])
                children.append(child1)
                children.append(child2)

            # for _ in range(num_players):
            #     child = self.reproduction(parents)
            #     children.append(child)

            return children

    def sus(self, players, num_players):
        # Create Intervals
        interval_length = 1 - 1 / num_players
        intervals = np.linspace(0, interval_length, num_players)
        random_number = np.random.uniform(0, 1 / num_players, 1)
        intervals += random_number

        probabilities = self.calculate_cumulative_probabilities(players)

        result = []
        for interval in intervals:
            for i, probability in enumerate(probabilities):
                if interval < probability:
                    result.append(copy.deepcopy(players[i]))
                    break

        return result

    def next_population_selection(self, players, num_players):
        # num_players example: 100
        # players: an array of `Player` objects
        result = players
        selection_method = "roulette wheel"

        if selection_method == "top-k":
            sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
            result = sorted_players[: num_players]
        elif selection_method == 'roulette wheel':
            result = self.roulette_wheel(players, num_players)
        elif selection_method == "SUS":
            result = self.sus(players, num_players)

        # (additional): plotting
        fitness_list = [player.fitness for player in players]
        max_fitness = float(np.max(fitness_list))
        mean_fitness = float(np.mean(fitness_list))
        min_fitness = float(np.min(fitness_list))
        self.save_fitness_result(min_fitness, max_fitness, mean_fitness)

        return result

    def save_fitness_result(self, min_fitness, max_fitness, mean_fitness):
        if self.generation_number == 0:
            fitness_results = {
                'min_fitness': [min_fitness],
                'max_fitness': [max_fitness],
                'mean_fitness': [mean_fitness]
            }
            with open('fitness_results.json', 'w') as out_file:
                json.dump(fitness_results, out_file)
        else:
            with open('fitness_results.json', 'r') as in_file:
                fitness_results = json.load(in_file)

            fitness_results['min_fitness'].append(min_fitness)
            fitness_results['max_fitness'].append(max_fitness)
            fitness_results['mean_fitness'].append(mean_fitness)

            with open('fitness_results.json', 'w') as out_file:
                json.dump(fitness_results, out_file)

        self.generation_number += 1
