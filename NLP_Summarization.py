from deap import algorithms, base, creator, tools
from nltk import word_tokenize
import re, string
import random

# class constants
num_words = 10  # number of words to print
n_gen = 150  # Increase to possibly improve best individual
pop_size = 250
prob_xover = 0.5
prob_mut = 0.4
tournament_size = 3
num_iterations = 10
dict_freq = {}  # frequency map for words in list
dict_paired_freq = {}


f = open('pride&prejudice_5_6_7.txt', 'r')
words = ''
for line in f:
    words += line

regex = re.compile('[,.;\"\'!?_-]')
words = regex.sub('', words)
tokenized_words = word_tokenize(words) # list of all words in new dictionary
WORD_LIST = list(set(tokenized_words)) # create a set out of words so there are no repeats in word list and make list again
dictionary_size = len(WORD_LIST)


# create a frequency dictionary of all the words in the text
for word in tokenized_words:
    if word not in dict_freq:
        dict_freq[word] = 1
    else:
        dict_freq[word] += 1

# create a frequency map of paired words in the text
for i in range(len(tokenized_words) - 1):
    if (tokenized_words[i+1],tokenized_words[i]) not in dict_paired_freq:
        dict_paired_freq[(tokenized_words[i+1],tokenized_words[i])] = 1
    else:
        dict_paired_freq[(tokenized_words[i+1],tokenized_words[i])] += 1

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def getFitness(individual):
    ind_sentence = ''
    fitness = 0
    for word_index in individual:
        ind_sentence += tokenized_words[word_index] + ' '
        if dict_freq[tokenized_words[word_index]] > 10:
            fitness -= 2
    tokenized_sentence = word_tokenize(ind_sentence)
    for i in range(len(tokenized_sentence) - 1):
        if (tokenized_sentence[i + 1], tokenized_sentence[i]) in dict_paired_freq:
            fitness += dict_paired_freq[(tokenized_sentence[i+1], tokenized_sentence[i])]
    return fitness,


toolbox = base.Toolbox()
# '6randomly placed words in a list ranging from 0-dictionary_size
toolbox.register("indices", random.sample, range(dictionary_size), num_words)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", getFitness)
toolbox.register("mate", tools.cxUniform, indpb=0.05)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=dictionary_size, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)

top_sentences = []

# Repeat algorithm ten times over 150 generations
for num_sentence in range(num_iterations):
    pop = toolbox.population(n=pop_size)
    bestfitlist = []
    bestfitval = 0;
    # Let's collect the fitness values from the simulation using
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(0, n_gen):
        # Start creating the children (or offspring)

        # First, Apply selection:
        offspring = toolbox.select(pop, pop_size)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply variations (xover and mutation), Ex: algorithms.varAnd(?, ?, ?, ?)
        offspring = algorithms.varAnd(offspring, toolbox, prob_xover, prob_mut)

        for fit in fitnesses:
            if bestfitval < fit[0]:
                bestfitval = fit[0]
        bestfitlist.append(bestfitval)

        # Let's evaluate the fitness of each individual.

        fitnesses = list(map(toolbox.evaluate, offspring))

    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    # One way of implementing elitism is to combine parents and children to give them equal chance to compete:
    # For example: pop[:] = pop + offspring
    # Otherwise you can select the parents of the generation from the offspring population only: pop[:] = offspring
    pop[:] = pop + offspring
    print("-- End of evolution --  Number of generations: " + str(n_gen))
    print("Best fitness values list: %s" % (bestfitlist))

    best_ind = tools.selBest(pop, 1)[0]
    best_ind_sentence = ''
    for i in best_ind:
        if i > dictionary_size:
            i = dictionary_size - 1
        best_ind_sentence += tokenized_words[i] + ' '

    display_sentence = "Sentence %s: \"%s\", best fitness: %s" % (num_sentence + 1, best_ind_sentence, bestfitlist[len(bestfitlist) - 1])
    print(display_sentence)
    top_sentences.append(display_sentence)

print("-- End of the %d evolutions --\n" % num_iterations)
best_ind = tools.selBest(pop, 1)[0]
print("The Top %d Sentences Generated:" % num_words)
for sentence in top_sentences:
    print(sentence)
