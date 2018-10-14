#!/bin/python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sns.set()
sns.set_context('paper')
sns.set_style('ticks')
sns.set_palette('Set2')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = [12, 9]

D = ['D_' + str(i) for i in range(10)]
columns = ['Tribe', 'Eval', 'fitness', 'sigma', 'var_sigma']
plot_names = ['Fitness', '$\sigma$', '$Var(\sigma)$']

df = pd.read_csv('data.csv', names=columns + D, skipfooter=2, engine='python')

for idx, column in enumerate(tqdm(columns[2:])):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale('log')
    df.groupby('Tribe')[column].plot(title=plot_names[idx], ax=ax)
    sns.despine()
    fig.savefig('{}.pgf'.format(column), dpi=200)
    fig.savefig('{}.pdf'.format(column), dpi=200)

fig = plt.figure()
for d in tqdm(D):
    ax = plt.gca()
    ax.set_yscale('log')
    df.groupby('Tribe')[d].plot(ax=ax)
    sns.despine()
fig.savefig('eigenvalues.pgf', dpi=200)
fig.savefig('eigenvalues.pdf', dpi=200)
