import matplotlib.pyplot as plt
import json

with open('fitness_results.json', 'r') as in_file:
    fitness_results = json.load(in_file)

number_of_generations = len(fitness_results['min_fitness'])
x = [i for i in range(number_of_generations)]

fig, ax = plt.subplots(nrows=3,figsize=(5, 10))
ax[0].plot(x, fitness_results['min_fitness'])
ax[0].set(title="Min Fitness")

ax[1].plot(x, fitness_results['max_fitness'])
ax[1].set(title="Max Fitness")

ax[2].plot(x, fitness_results['mean_fitness'])
ax[2].set(title="Mean Fitness")

fig.tight_layout(pad=3)

plt.show()

