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
    return pd.DataFrame(a)
def single_run(function, seed, _lambda):
    fitnesses = []
    cond = []
    sigmas = []
    normps = []
    result = subprocess.run(['java', '-Dverbose=true', '-Dlambda={}'.format(_lambda), '-jar', 'testrun.jar', '-submission=player28', '-evaluation={}'.format(function), '-seed={}'.format(seed)], stdout=PIPE, stderr=PIPE, encoding='UTF-8')
    for line in result.stdout.split('\n'):
        line = line.rstrip();
        if line[:5] == "Score":
            break
        sline = line.split()
        if len(sline) < 14:
            break
        try:
            fitnesses.append(float(sline[5]))
            sigmas.append(float(sline[8]))
            cond.append(float(sline[11]))
            normps.append(float(sline[14]))
        except ValueError as e:
            break
    return fitnesses, sigmas, normps, cond

function = functions[2]
_lambda = 195
fitnesses = []
sigmas = []
normps = []
cond = []
gens = []
seeds = []
for _ in tqdm(range(0, 10), desc="{}".format(_lambda), leave=False):
    seed = random.randint(1,10000000)
    # seed = 34032434
    _fitnesses, _sigmas, _normps, _cond = single_run(function, seed, _lambda)
    fitnesses.append(_fitnesses)
    sigmas.append(_sigmas)
    normps.append(_normps)
    cond.append(_cond)
    gens.append(len(_fitnesses))
    seeds.append(seed)
fitnesses = padded_sum(fitnesses)
sigmas = padded_sum(sigmas)
normps = padded_sum(normps)
cond = padded_sum(cond)
gens = np.array(gens)
seeds = np.array(seeds)

fig, ax = plt.subplots(nrows=2, ncols=2)
fitnesses.plot(ax=ax[0,0], title="Maximum Fitness")
sigmas.plot(ax=ax[0,1], title="Step size")
normps.plot(ax=ax[1,0], title="Norm of P_sigma")
cond.plot(ax=ax[1,1], title="Conditioning")
plt.show()
