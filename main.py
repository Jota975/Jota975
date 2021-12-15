import random as r
from datetime import datetime
import matplotlib.pyplot as plt
import copy
import math
import numpy as np

file = open("results4.txt", "w")

N = 20
P = 100
PARENTS = # (tournament size) if using tournament selection 4,8,12
EPOCH = 200
MAX_RANGE = # depending on the fitness function used
MIN_RANGE = # depending on the fitness function used
RANGE = MAX_RANGE * 2

population = []
totalfitness = []
r.seed(datetime.now())


# gene class
class individual:
    gene = 0 * [N]
    fitness = 0


# fitness functions
def fitnesscheck(ind):
    fitness = 0

    for f in range(N):
        fitness = fitness + ind.gene[f]
    return round(fitness, 3)


def fitnesscheck2(ind):
    fitness = 0

    for f in range(N):
        fitness = fitness + ((ind.gene[f]) ** 2) - (10 * math.cos(2 * math.pi * ind.gene[f]))

    fitness = fitness + 10 * N

    return round(fitness, 3)


def fitnesscheck3(ind):
    fitness = 0

    for f in range(N - 1):
        fitness = fitness + (100 * (ind.gene[f + 1] - (ind.gene[f] ** 2)) ** 2 + ((1 - ind.gene[f]) ** 2))

    return round(fitness, 3)


def fitnesscheck4(ind):
    fitness = 0
    sum1 = 0
    sum2 = 0

    for f in range(N):
        sum1 = sum1 + (ind.gene[f] ** 2)
        sum2 = sum2 + (math.cos(2 * math.pi * ind.gene[f]))

    sum1 = -20 * (math.exp(-0.2 * math.sqrt(sum1 / N)))
    sum2 = math.exp(sum2 / N)
    fitness = sum1 - sum2

    return fitness


mrs = np.arange(0.1, 0.5, 0.01)
mss = np.arange(MAX_RANGE/10, MAX_RANGE/2, 0.01*RANGE)

AVG = 20
run = 0

for mr in range(1): #changed for the mutation rate/step search



    for ms in range(1): #changed for the mutation rate/step search

        #file.write("%.2f  " % (mrs[mr]))
        #file.write("%.2f  " % (mss[ms]))

        averagemeans = []
        averagefitest = []
        avrgbestcand = []
        finalavg = []
        finalfit = []
        gentakenavg = 0
        bestfitavg = 0

        for avrg in range(AVG): # average for loop
            MUTRATE = # define mutation rate
            MUTSTEP = # define mutation step

            mutup = 0
            mutdown = 0

            meanfitness = []
            genfitest = []

            # create and test first population
            population = []
            for x in range(P):
                tempgene = []

                for x in range(0, N):
                    randgene = round(r.uniform(MIN_RANGE, MAX_RANGE), 3)
                    tempgene.append(randgene)

                newind = individual()
                newind.gene = tempgene.copy()
                population.append(newind)

            generationf = 0
            for p in range(0, P):
                population[p].fitness = fitnesscheck3(population[p])
                generationf = generationf + population[p].fitness

            totalfitness.append(generationf)
            meanfit = round(generationf / P, 3)
            meanfitness.append(meanfit)

            bestindex = 0

            for x in range(EPOCH):
                temppop = []
                offsprings = []

                # best gene population
                fitestgene = N ** 10
                for fitest in range(P):
                    if population[fitest].fitness < fitestgene:
                        fitestgene = population[fitest].fitness
                        bestindex = fitest

                genfitest.append(population[bestindex].fitness)

                """"
                # dynamic mutation rate/step
                if x > 1:
                    if genfitest[x - 1] == genfitest[x]:
                        mutup = mutup + 1
                        mutdown = 0
                    if genfitest[x] < genfitest[x - 1]:
                        mutdown = mutdown + 1
                        mutup = 0

                if mutup >= 5:
                    MUTRATE = MUTRATE + 0.01
                    MUTSTEP = MUTSTEP + 0.01*RANGE
                    mutup = 0

                if mutdown >= 1:
                    MUTRATE = MUTRATE - 0.01
                    MUTSTEP = MUTSTEP - 0.01 * RANGE
                    mutdown = 0
                """

                # parent selection

                # tournament selection
                """
                for parents in range(P):
                  parentparticipants = []
                  for participant in range(PARENTS):
                    parentparticipants.append(population[r.randint(0, P-1)])
              
                  winnerparent = copy.copy(parentparticipants[0])
                  for tournament in range(PARENTS):
                    if parentparticipants[tournament].fitness < winnerparent.fitness:
                      winnerparent = parentparticipants[tournament]
              
                  temppop.append(winnerparent)
                """

                # roullete wheel selection
                parent_prob = []
                for p in range(P):
                    rfit = 1 - (population[p].fitness / generationf)
                    parent_prob.append(rfit)

                temppop = np.random.choice(population, P, parent_prob)

                # recombination

                # single point crossover

                toff1 = individual()
                toff2 = individual()
                temp = individual()
                for i in range(0, P, 2):
                    toff1 = copy.deepcopy(temppop[i])
                    toff2 = copy.deepcopy(temppop[i + 1])
                    temp = copy.deepcopy(temppop[i])
                    crosspoint = r.randint(1, N)
                    for j in range(crosspoint, N):
                        toff1.gene[j] = toff2.gene[j]
                        toff2.gene[j] = temp.gene[j]
                    temppop[i] = copy.deepcopy(toff1)
                    temppop[i + 1] = copy.deepcopy(toff2)

                """
                # uniform crossover
                toff1 = individual()
                toff2 = individual()
                temp = individual()
                for i in range(0, P, 2):
                    toff1 = copy.deepcopy(temppop[i])
                    toff2 = copy.deepcopy(temppop[i + 1])
                    temp = copy.deepcopy(temppop[i])

                    for j in range(N):
                        if r.randint(0, 1) == 1:
                            toff1.gene[j] = toff2.gene[j]
                            toff2.gene[j] = temp.gene[j]

                    temppop[i] = copy.deepcopy(toff1)
                    temppop[i + 1] = copy.deepcopy(toff2)
                """

                # mutation
                for m in range(0, P):
                    for j in range(0, N):
                        if r.randint(0, 100) < (100 * MUTRATE):
                            alter = round(r.uniform(0, MUTSTEP), 3)
                            if r.randint(0, 1) == 1:
                                temppop[m].gene[j] = temppop[m].gene[j] + alter
                                if temppop[m].gene[j] > MAX_RANGE:
                                    temppop[m].gene[j] = MAX_RANGE
                            else:
                                temppop[m].gene[j] = temppop[m].gene[j] - alter
                                if temppop[m].gene[j] < MIN_RANGE:
                                    temppop[m].gene[j] = MIN_RANGE

                    offsprings.append(temppop[m])

                # calculate offsprings fitness
                generationf = 0

                for f in range(0, P):
                    offsprings[f].fitness = fitnesscheck3(offsprings[f])
                    generationf = generationf + offsprings[f].fitness

                totalfitness.append(generationf)
                meanfit = round(generationf / P, 3)
                meanfitness.append(meanfit)

                # worst offspring
                worstindex = 0
                worstgene = population[0].fitness
                for worst in range(0, P):
                    if offsprings[worst].fitness > worstgene:
                        worstgene = offsprings[worst].fitness
                        worstindex = worst

                # replace new worst with last best
                offsprings[worstindex] = population[bestindex]

                # population offspring copy
                population = copy.copy(offsprings)

            # best gene population
            fitestgene = N ** 10
            for fitest in range(P):
                if population[fitest].fitness < fitestgene:
                    fitestgene = population[fitest].fitness
                    bestindex = fitest

            gentaken = genfitest.index(genfitest[EPOCH - 1])
            gentakenavg = gentakenavg + gentaken

            # print(population[bestindex].gene)
            bestfitavg = round(bestfitavg + population[bestindex].fitness, 3)
            avrgbestcand.append(population[bestindex])

            """
            plt.plot(meanfitness)
            plt.plot(genfitest)
            plt.show()
            """

            averagemeans.append(meanfitness)
            averagefitest.append(genfitest)

        gentakenavg = round(gentakenavg / AVG, 3)
        bestfitavg = round(bestfitavg / AVG, 3)

        #file.write("%.2f  " % gentakenavg)
        #file.write("%.2f  " % bestfitavg)
        print(gentakenavg)
        print(bestfitavg)

        # calculating avrg mean fitness over AVG runs
        for e in range(EPOCH + 1):
            sum1 = 0
            for av in range(AVG):
                sum1 = sum1 + averagemeans[av][e]

            finalavg.append(sum1 / AVG)

        # calculating avrg best fitness over AVG runs
        for e in range(EPOCH):
            sum1 = 0
            for av in range(AVG):
                sum1 = sum1 + averagefitest[av][e]

            finalfit.append(sum1 / AVG)

        s = 0
        for c in range(AVG):
            s = s + fitnesscheck(avrgbestcand[c])
        s = round(s / AVG, 3)
        #file.write("%.2f  " % s)
        print(s)

        #file.write("\n")


        plt.plot(finalavg)
        plt.plot(finalfit)
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
        plt.legend(['Mean Fitness', 'Best Fitness'])
        plt.show()

