#!/usr/bin/env python3
#
# Figure 3: Error vs evaluations & run-time, for IKr
#
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

import methods

poster = 'poster' in sys.argv

c1 = '#fdae61'
c2 = '#d7191c'
c3 = '#92c5de'
c4 = '#2c7bb6'

matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# Create figure
print('Creating figure')
if poster:
    fig = plt.figure(figsize=(9, 4.5))
    fig.subplots_adjust(0.075, 0.095, 0.99, 0.99, wspace=0.04, hspace=0.25)
else:
    fig = plt.figure(figsize=(4.5, 3.5))  # One-column CINC size
    fig.subplots_adjust(0.145, 0.120, 0.99, 0.99, wspace=0.04, hspace=0.40)

grid = fig.add_gridspec(2, 2)

ik_cma = methods.fitio.load_logs('results/ikr-cmaes/result.txt')
ik_rpr = methods.fitio.load_logs('results/ikr-rprop/result.txt')

a = 0.5
ds = 'steps-pre'
ev_xlim = 0, 17000
rt_xlim = 0, 3250
ylim = 4e-4, 2e4

# Top-left: CMA-ES evaluations
ax00 = fig.add_subplot(grid[0, 0])
ax00.set_xlabel('Evaluations')
ax00.set_ylabel('IKr Error')
ax00.set_yscale('log')
ax00.set_xlim(*ev_xlim)
ax00.set_ylim(*ylim)
if not poster:
    ax00.set_xticks([0, 5000, 10000, 15000])
    ax00.set_xticklabels(['0', '5k', '10k', '15k'])
    ax00.set_yticks([1e-3, 1, 1e3])
la = 'CMA-ES'
for e, f, t in ik_cma:
    ax00.plot(e, f, color=c1, ds=ds, alpha=a, label=la)
    la = None
la = 'rProp-'
for e, f, t in ik_rpr:
    ax00.plot(e, f, color='#cccccc', ds=ds, alpha=a, label=la, zorder=0)
    la = None
ax00.legend(loc='upper right', frameon=poster)

# Top-right: rProp- evaluations
ax01 = fig.add_subplot(grid[0, 1])
ax01.set_xlabel('Evaluations')
#ax01.set_ylabel('IKr Error')
ax01.set_yscale('log')
ax01.set_yticklabels([])
ax01.set_xlim(*ev_xlim)
ax01.set_ylim(*ylim)
if not poster:
    ax01.set_xticks([0, 5000, 10000, 15000])
    ax01.set_xticklabels(['0', '5k', '10k', '15k'])
    ax01.set_yticks([1e-3, 1, 1e3])
la = 'rProp-'
for e, f, t in ik_rpr:
    ax01.plot(e, f, color=c2, ds=ds, alpha=a, label=la)
    la = None
la = 'CMA-ES'
for e, f, t in ik_cma:
    ax01.plot(e, f, color='#cccccc', ds=ds, alpha=a, label=la, zorder=0)
    la = None
ax01.legend(loc='upper right', frameon=poster)

# Bottom-left: CMA-ES run times
ax10 = fig.add_subplot(grid[1, 0])
ax10.set_xlabel('Run time (s)')
ax10.set_ylabel('IKr Error')
ax10.set_yscale('log')
ax10.set_xlim(*rt_xlim)
ax10.set_ylim(*ylim)
if not poster:
    ax10.set_xticks([0, 1000, 2000, 3000])
    ax10.set_xticklabels(['0', '1k', '2k', '3k'])
    ax10.set_yticks([1e-3, 1, 1e3])
la = 'CMA-ES'
for e, f, t in ik_cma:
    ax10.plot(t, f, color=c1, ds=ds, alpha=a, label=la)
    la = None
la = 'rProp-'
for e, f, t in ik_rpr:
    ax10.plot(t, f, color='#cccccc', ds=ds, alpha=a, label=la, zorder=0)
    la = None
ax10.legend(loc='upper right', frameon=poster)

# Bottom-right: rProp- run times
ax11 = fig.add_subplot(grid[1, 1])
ax11.set_xlabel('Run time (s)')
#ax11.set_ylabel('IKr Error')
ax11.set_yscale('log')
ax11.set_yticklabels([])
ax11.set_xlim(*rt_xlim)
ax11.set_ylim(*ylim)
if not poster:
    ax11.set_xticks([0, 1000, 2000, 3000])
    ax11.set_xticklabels(['0', '1k', '2k', '3k'])
    ax11.set_yticks([1e-3, 1, 1e3])
la = 'rProp-'
for e, f, t in ik_rpr:
    ax11.plot(t, f, color=c2, ds=ds, alpha=a, label=la)
    la = None
la = 'CMA-ES'
for e, f, t in ik_cma:
    ax11.plot(t, f, color='#cccccc', ds=ds, alpha=a, label=la, zorder=0)
    la = None
ax11.legend(loc='upper right', frameon=poster)

# Finalise & store
fig.align_ylabels([ax00, ax10])
fig.align_ylabels([ax01, ax11])

if poster:
    path = os.path.join('figures', 'p3-opt-ikr.pdf')
else:
    path = os.path.join('figures', 'f3-opt-ikr.pdf')
print(f'Writing figure to {path}')
plt.savefig(path)

