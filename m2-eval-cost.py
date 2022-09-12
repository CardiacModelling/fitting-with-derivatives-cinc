#!/usr/bin/env python3
#
# Figure 2: Run time per evaluation
#
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import myokit
import numpy as np

poster = 'poster' in sys.argv

c1 = '#fdae61'
c2 = '#d7191c'
c3 = '#92c5de'
c4 = '#2c7bb6'

matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# Load data
ik = myokit.DataLog.load_csv(os.path.join('results', 'eval-cost-ikr.csv'))
ap = myokit.DataLog.load_csv(os.path.join('results', 'eval-cost-ap.csv'))
ik, ap = ik.npview(), ap.npview()

ikc = ik['count']
ik1 = ik['without_derivatives_seconds']
ik2 = ik['with_derivatives_seconds']

apc = ap['count']
ap1 = ap['without_derivatives_seconds']
ap2 = ap['with_derivatives_seconds']

# Create figure
print('Creating figure')
if poster:
    fig = plt.figure(figsize=(9, 2.5))
    fig.subplots_adjust(0.055, 0.17, 0.99, 0.98, wspace=0.8, hspace=0.25)
    grid = fig.add_gridspec(1, 6)
else:
    fig = plt.figure(figsize=(4.5, 3.5))  # One-column CINC size
    fig.subplots_adjust(0.130, 0.120, 0.99, 0.98, wspace=1.3, hspace=0.40)
    grid = fig.add_gridspec(2, 4)

ax00 = fig.add_subplot(grid[0, :3])
ax00.set_xlabel('IKr error function evaluations')
ax00.set_ylabel('Run time (s)')
ax00.plot(ikc, np.cumsum(ik1), color=c1, label='No derivatives')
ax00.plot(ikc, np.cumsum(ik2), color=c2, label='With derivatives')
ax00.legend(loc='upper left', frameon=poster)
ax00.set_xlim(0, 200)
ax00.set_ylim(0, 42)

ax10 = fig.add_subplot(grid[0, 3:5] if poster else grid[1, :3])
ax10.set_xlabel('AP error function evaluations')
ax10.set_ylabel('Run time (s)')
ax10.plot(apc, np.cumsum(ap1), color=c3, label='No derivatives')
ax10.plot(apc, np.cumsum(ap2), color=c4, label='With derivatives')
ax10.legend(loc='upper left', frameon=poster)
ax10.set_xlim(0, 200)
ax10.set_ylim(0, 170)

ax01 = fig.add_subplot(grid[0, 2] if poster else grid[0, 3])
ax01.set_ylabel('Mean time (s)')
ax01.bar([1], [np.mean(ik1)], color=c1)
ax01.bar([2], [np.mean(ik2)], color=c2)
ax01.set_ylim(0, 0.82)
ax01.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

ax11 = fig.add_subplot(grid[-1])
ax11.set_ylabel('Mean time (s)')
ax11.bar([1], [np.mean(ap1)], color=c3)
ax11.bar([2], [np.mean(ap2)], color=c4)
#ax11.set_ylim(0, 0.82)
ax11.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

print()
print('IKr')
print(f'Mean, free: {np.mean(ik1)}')
print(f'Mean, sens: {np.mean(ik2)}')
print(f'Ratio: {np.mean(ik2) / np.mean(ik1)}')

print()
print('AP')
print(f'Mean, free: {np.mean(ap1)}')
print(f'Mean, sens: {np.mean(ap2)}')
print(f'Ratio: {np.mean(ap2) / np.mean(ap1)}')


# Finalise & store
fig.align_ylabels([ax00, ax10])
#fig.align_ylabels([ax01, ax11])

if poster:
    path = os.path.join('figures', 'p2-eval-cost.pdf')
else:
    path = os.path.join('figures', 'f2-eval-cost.pdf')
print(f'Writing figure to {path}')
plt.savefig(path)
