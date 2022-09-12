#!/usr/bin/env python3
#
# Figure 5: Robustness
#
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import methods

# TODO: Make version for paper
poster = 'poster' in sys.argv

c1 = '#fdae61'
c2 = '#d7191c'
c3 = '#92c5de'
c4 = '#2c7bb6'

matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def robustness(parameters, info):
    """ Returns a tuple ``(e, d, ne, np)`` for a robustness plot. """

    # info = (run, error, time, iterations, evaluations)
    # Extract errors
    e = info[:, 1]
    e = e / e[0] - 1

    # Extract max distance in parameters
    d = np.max(np.abs(parameters / parameters[0] - 1), axis=1)

    # Count how many scores were within 5% of best
    eb = e[e < 0.05]

    # Count number near best
    db = d[d < 0.05]

    return e, d, len(eb), len(db)


ik_cma = robustness(*methods.fitio.load('results/ikr-cmaes/result.txt', 9))
ik_rpr = robustness(*methods.fitio.load('results/ikr-rprop/result.txt', 9))
ap_cma = robustness(*methods.fitio.load('results/ap-cmaes/result.txt', 10))
ap_rpr = robustness(*methods.fitio.load('results/ap-rprop/result.txt', 10))

# Create figure
print('Creating figure')
if poster:
    fig = plt.figure(figsize=(9, 3.7))
    fig.subplots_adjust(0.075, 0.115, 0.98, 0.99, wspace=0.25, hspace=0.25)
    grid = fig.add_gridspec(1, 2)
else:
    fig = plt.figure(figsize=(4.5, 6))  # One-column CINC size
    fig.subplots_adjust(0.150, 0.075, 0.97, 0.99, hspace=0.20)
    grid = fig.add_gridspec(2, 1)

sub1 = grid[0].subgridspec(2, 2, wspace=0.15)
sub2 = grid[1].subgridspec(2, 2, wspace=0.15)

# IKr Error
xlim = 0, 100
ylim = 8e-12, 5e4
ax00 = fig.add_subplot(sub1[0, 0])
ax00.set_ylabel('$(E - E_0) / E_0$')
ax00.set_yscale('log')
ax00.set_xlim(*xlim)
ax00.set_ylim(*ylim)
ax00.plot(ik_cma[0], color=c1)

ax01 = fig.add_subplot(sub1[0, 1])
ax01.set_yscale('log')
ax01.set_yticklabels([])
ax01.set_xlim(*xlim)
ax01.set_ylim(*ylim)
ax01.plot(ik_rpr[0], color=c2)

# IKr parameters
ylim = 5e-7, 1e8
ax10 = fig.add_subplot(sub1[1, 0])
ax10.set_xlabel('Index')
ax10.set_ylabel('$\max |p_i - p_{i,0}| / p_{i,0}$')
ax10.set_yscale('log')
ax10.set_xlim(*xlim)
ax10.set_ylim(*ylim)
ax10.plot(ik_cma[1], color=c1)

ax11 = fig.add_subplot(sub1[1, 1])
ax11.set_xlabel('Index')
ax11.set_yscale('log')
ax11.set_yticklabels([])
ax10.set_xlim(*xlim)
ax11.set_ylim(*ylim)
ax11.plot(ik_rpr[1], color=c2)

# AP Error
ylim = 5e-10, 5e2
ax02 = fig.add_subplot(sub2[0, 0])
ax02.set_ylabel('$(E - E_0) / E_0$')
ax02.set_yscale('log')
ax02.set_xlim(*xlim)
ax02.set_ylim(*ylim)
ax02.plot(ap_cma[0], color=c3)

ax03 = fig.add_subplot(sub2[0, 1])
ax03.set_yscale('log')
ax03.set_yticklabels([])
ax03.set_xlim(*xlim)
ax03.set_ylim(*ylim)
ax03.plot(ap_rpr[0], color=c4)

# AP parameters
ylim = 1e-6, 10
ax12 = fig.add_subplot(sub2[1, 0])
ax12.set_xlabel('Index')
ax12.set_ylabel('$\max |p_i - p_{i,0}| / p_{i,0}$')
ax12.set_yscale('log')
ax12.set_xlim(*xlim)
ax12.set_ylim(*ylim)
ax12.plot(ap_cma[1], color=c3)

ax13 = fig.add_subplot(sub2[1, 1])
ax13.set_xlabel('Index')
ax13.set_yscale('log')
ax13.set_yticklabels([])
ax13.set_xlim(*xlim)
ax13.set_ylim(*ylim)
ax13.plot(ap_rpr[1], color=c4)

# Labels
ax00.text(0.1, 0.8, 'IKr, CMA-ES', transform=ax00.transAxes)
ax10.text(0.1, 0.8, 'IKr, CMA-ES', transform=ax10.transAxes)
ax01.text(0.1, 0.8, 'IKr, rProp-', transform=ax01.transAxes)
ax11.text(0.1, 0.8, 'IKr, rProp-', transform=ax11.transAxes)
ax02.text(0.1, 0.8, 'AP, CMA-ES', transform=ax02.transAxes)
ax12.text(0.1, 0.8, 'AP, CMA-ES', transform=ax12.transAxes)
ax03.text(0.1, 0.8, 'AP, rProp-', transform=ax03.transAxes)
ax13.text(0.1, 0.8, 'AP, rProp-', transform=ax13.transAxes)

grey = '#f2f2f2'
ax00.axvspan(0, ik_cma[2], color=grey)
ax10.axvspan(0, ik_cma[3], color=grey)
ax01.axvspan(0, ik_rpr[2], color=grey)
ax11.axvspan(0, ik_rpr[3], color=grey)
ax02.axvspan(0, ap_cma[2], color=grey)
ax12.axvspan(0, ap_cma[3], color=grey)
ax03.axvspan(0, ap_rpr[2], color=grey)
ax13.axvspan(0, ap_rpr[3], color=grey)

dark = '#999999'
ax00.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax10.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax01.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax11.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax02.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax12.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax03.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)
ax13.axhline(0.05, ls='--', lw=1, color=dark, zorder=1)

# Finalise & store
fig.align_ylabels([ax00, ax10])

if poster:
    path = os.path.join('figures', 'p5-robustness.pdf')
else:
    path = os.path.join('figures', 'f5-robustness.pdf')

print(f'Writing figure to {path}')
plt.savefig(path)

