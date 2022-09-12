#!/usr/bin/env python3
#
# Count fitting results
#
import os

import methods

cases = methods.Case.names()
meths = methods.Optimiser.names()

# Gather contents
rows = []
row = [''] + cases
rows.append(row)
for method in meths:
    row = [method]
    for case in cases:
        name = f'{case}-{method}'
        path = os.path.join('results', name, 'result.txt')
        row.append(str(methods.fitio.count(path, 1)))
    rows.append(row)

# Gather column widths
ncol = [0] * len(rows[0])
for row in rows:
    for i, col in enumerate(row):
        ncol[i] = max(len(col) + 1, ncol[i])

# Print table
for row in rows:
    row = [x + ' ' * (w - len(x)) for x, w in zip(row, ncol)]
    print('| ' + ' | '.join(row) + ' |')

