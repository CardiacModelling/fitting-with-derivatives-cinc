#!/usr/bin/env python3
#
# Figure 1: Test cases
#
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import methods

poster = 'poster' in sys.argv or True

c1 = '#fdae61'
c2 = '#d7191c'
c3 = '#92c5de'
c4 = '#2c7bb6'
cd = '#888888'

matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# Create figure
print('Creating figure')
if poster:
    fig = plt.figure(figsize=(9, 5.5))
    fig.subplots_adjust(0.085, 0.08, 0.99, 0.99, wspace=0.25, hspace=0.10)
grid = fig.add_gridspec(4, 2)

# IKr
ax00 = fig.add_subplot(grid[0, 0])
ax10 = fig.add_subplot(grid[1:3, 0])
ax20 = fig.add_subplot(grid[3, 0])

for ax in (ax00, ax10, ax20):
    ax.set_xlim(0, 15.400)

ax00.set_xticklabels([])
ax10.set_xticklabels([])
ax20.set_xlabel('Time (s)')

ax00.set_ylabel('V (mV)')
ax10.set_ylabel('I (A/F)')
ax20.set_ylabel('dI / dp$_0$')

ax10.set_ylim(-0.25, 1.55)
ax20.set_ylim(-150, 850)

if True:
    ik_model = methods.ikr.ModelWithSensitivities()
    d1, s1 = ik_model.sim.run(15399)
    d1 = d1.npview()
    s1 = np.array(s1)
    d1['engine.time'] *= 1e-3

    ax00.plot(d1.time(), d1['membrane.V'], color=cd, ds='steps-post')
    ax10.plot(d1.time(), d1['ikr.IKr'], color=c2)
    ax20.plot(d1.time(), s1[:, 0, 0], color=cd)

# AP
ax01 = fig.add_subplot(grid[0, 1])
ax11 = fig.add_subplot(grid[1:3, 1])
ax21 = fig.add_subplot(grid[3, 1])

for ax in (ax01, ax11, ax21):
    ax.set_xlim(0, 9.050)

ax01.set_xticklabels([])
ax11.set_xticklabels([])
ax21.set_xlabel('Time (s)')

ax01.set_ylabel('V (mV)')
ax11.set_ylabel('I (A/F)')
ax21.set_ylabel('dI / dg$_f$')

ax11.set_ylim(-9, 3)
#ax21.set_ylim(-0.02, 0.001)

if True:
    ap_model = methods.ap.ModelWithSensitivities()
    d1, s1 = ap_model.sim.run(9050)
    d1 = d1.npview()
    s1 = np.array(s1)
    d1['engine.time'] *= 1e-3

    ax01.plot(d1.time(), d1['membrane.V'], color=cd)
    ax11.plot(d1.time(), d1['membrane.i_ion'], color=c4)
    ax21.plot(d1.time(), s1[:, 0, 8], color=cd)

    # Bands
    events = [e for e in ap_model._protocol]
    events = [events[i] for i in (0, 4, 7, 10, 13, 17, 21)]
    times = np.array([e.start() for e in events] + [9050]) * 1e-3
    names = ['$I_{Kr}$', '$I_{CaL}$', '$I_{Na}$', '$I_{to}$', '$I_{K1}$',
             '$I_f$', '$I_{Ks}$']
    for i in range(len(times) - 1):
        t1, t2 = times[i], times[1 + i]
        y = -65 if i in (4, ) else -110
        ax01.text(0.5 * (t1 + t2), y, names[i], ha='center')
        if i % 2:
            ax01.axvspan(t1, t2, color='#f0f0f0f0', zorder=0)
            ax11.axvspan(t1, t2, color='#f0f0f0f0', zorder=0)
            ax21.axvspan(t1, t2, color='#f0f0f0f0', zorder=0)

# Finalise & store
fig.align_ylabels([ax00, ax10, ax20])
fig.align_ylabels([ax01, ax11, ax21])

if poster:
    path = os.path.join('figures', 'p1-cases.pdf')
print(f'Writing figure to {path}')
plt.savefig(path)

