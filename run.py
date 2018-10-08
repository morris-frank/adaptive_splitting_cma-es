#!/bin/python
import subprocess
from subprocess import PIPE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

sns.set_style("whitegrid")

functions = ['BentCigarFunction', 'SchaffersEvaluation', 'KatsuuraEvaluation']
command = 'java -Dlambda={lambda} -jar testrun.jar -submission=player28 -evaluation={func} -seed={seed}'

def padded_sum(a):
    max_len = len(max(a, key=len))
    nL = len(a)
    for i in range(nL):
        for _ in range(max_len - len(a[i])):
            a[i] = a[i] + [a[i][-1]]
    a = np.array(a).sum(axis=0) / nL
    return a

def single_run(function, seed, _lambda):
    fitnesses = []
    sigmas = []
    result = subprocess.run(['java', '-Dverbose=true', '-Dlambda={}'.format(_lambda), '-jar', 'testrun.jar', '-submission=player28', '-evaluation={}'.format(function), '-seed={}'.format(seed)], stdout=PIPE, stderr=PIPE, encoding='UTF-8')
    for line in result.stdout.split('\n'):
        line = line.rstrip();
        if line[:5] == "Score":
            break
        sline = line.split()
        # generation = int(line[4:9])
        try:
            fitnesses.append(float(sline[5]))
            sigmas.append(float(sline[8]))
        except ValueError as e:
            break
    return fitnesses, sigmas

def sign_single_run(function, _lambda):
    fitnesses = []
    sigmas = []
    maxgens = []
    maxfits = []
    for _ in tqdm(range(0, 30), desc="{}".format(_lambda), leave=False):
        seed = random.randint(1,10000000)
        _fitnesses, _sigmas = single_run(function, seed, _lambda)
        maxfits.append(max(_fitnesses))
    return np.mean(maxfits)
    #     fitnesses.append(_fitnesses)
    #     sigmas.append(_sigmas)
    # fitnesses = padded_sum(fitnesses)
    # sigmas = padded_sum(sigmas)
    # plt.plot(sigmas)
    # plt.show()

def lambda_test(function):
    maxfits = []
    lambda_range = range(10, 200, 5)
    for _lambda in tqdm(lambda_range, desc="{}".format(function)):
        maxfits.append(sign_single_run(function, _lambda))
    plt.scatter(lambda_range, maxfits, c="g", marker="X")
    plt.savefig('{}_maxfits_mean.png'.format(function), dpi=300)



def main():
    lambda_test(functions[0])

if __name__ == '__main__':
    main()
