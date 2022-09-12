#!/usr/bin/env python3
#
# Show best fitting results per problem & optimiser
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
        n_p = methods.Case.from_name(case).n_parameters()
        _, info = methods.fitio.load(path, n_p)
        if len(info):
            row.append(str(info[0][1]))
        else:
            row.append('')
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

