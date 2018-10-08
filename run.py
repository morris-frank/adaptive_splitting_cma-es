#!/bin/python
import subprocess
from subprocess import PIPE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

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
        fitnesses.append(float(sline[5]))
        sigmas.append(float(sline[8]))
    return fitnesses, sigmas

def sign_single_run(function, _lambda):
    fitnesses = []
    sigmas = []
    maxgens = []
    maxfits = []
    for _ in tqdm(range(0, 5), desc="{}".format(_lambda)):
        seed = random.randint(1,100)
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
    for _lambda in range(90, 200, 20):
        maxfits.append(sign_single_run(function, _lambda))
    plt.scatter(range(90, 200, 20), maxfits)
    plt.savefig('{}_maxfits_mean.png'.format(function), dpi=300)



def main():
    lambda_test(functions[2])

if __name__ == '__main__':
    main()
