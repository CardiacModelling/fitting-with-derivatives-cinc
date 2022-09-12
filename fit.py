#!/usr/bin/env python3
import os

import methods


# Get problem info
path, case, method = methods.fitio.fit_cmd('Fit')

# Show user what's happening
print('=' * 79)
print(path)
print('=' * 79)

# Try fitting
methods.fitio.fit(path, case, method, 100, 100)

# Show current best results
parameters, info = methods.fitio.load(
    os.path.join(path, 'result.txt'),
    n_parameters=case.n_parameters(),
)
methods.fitio.show_summary(parameters, info)
