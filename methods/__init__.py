#!/usr/bin/env python3
#
# Shared code for inference-with-sensitivities project.
#
import os
import inspect
frame = inspect.currentframe()
DIR = os.path.abspath(os.path.join(
    os.path.dirname(inspect.getfile(frame)), '..'))
del os, inspect, frame

# Test cases
from ._config import (  # noqa
    Case,
    Optimiser,
)

# Utilities modules
from . import (  # noqa
    fitio,
)

# Cases
from . import (  # noqa
    ap,
    ikr,
)

