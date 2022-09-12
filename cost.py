#!/usr/bin/env python3
#
# Evaluate errors and store the time it took
#
import argparse
import os
import sys

import myokit
import numpy as np

import methods

n = 200

# Parse command line arguments
parser = argparse.ArgumentParser('Cost')
parser.add_argument(
    'case',
    nargs='?',
    choices=methods.Case.names(),
    default='ikr',
    help='The test case to run')
args = parser.parse_args()
case = methods.Case.from_name(args.case)
path = os.path.join('results', f'eval-cost-{args.case}.csv')

# Load previous evaluations
print(f'Reading/writing {path}')
if os.path.isfile(path):
    d = myokit.DataLog.load_csv(path)
else:
    d = myokit.DataLog()
    d['count'] = []
    d['without_derivatives_seconds'] = []
    d['with_derivatives_seconds'] = []
t1s = d['without_derivatives_seconds']
t2s = d['with_derivatives_seconds']
assert len(t1s) == len(t2s)

# Run more
if len(t1s) < n:
    print(f'Found {len(t1s)} entries: adding another {n - len(t1s)}.')

    print('Compiling sims...')
    _, _, e1, b1, _, _ = case.build(False)
    _, _, e2, _, _, _ = case.build(True)
    print('Done')

    try:
        t = myokit.tools.Benchmarker()
        for i in range(len(t1s), n):
            print('.', end='')
            sys.stdout.flush()

            np.random.seed(i)
            p = b1.sample()[0]

            t.reset()
            e1(p)
            t1s.append(t.time())

            t.reset()
            e2(p)
            t2s.append(t.time())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print()

        n1, n2 = len(t1s), len(t2s)
        if n1 != n2:
            n1 = n2 = min(n1, n2)
            t1s = t1s[:n1]
            t2s = t2s[:n1]

        d['count'] = list(range(1, 1 + n1))
        d['without_derivatives_seconds'] = t1s
        d['with_derivatives_seconds'] = t2s

        print(f'Storing {n1} evaluations to {path}')
        d.save_csv(path)


print(f'Done ({len(t1s)} out of {n} evaluations).')
