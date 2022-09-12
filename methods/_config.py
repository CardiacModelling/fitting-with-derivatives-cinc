#!/usr/bin/env python3
#
# Named objects
#
import enum

import pints

import methods


class Case(enum.Enum):
    """
    Enumeration type for test cases.
    """

    ap = 'ap'
    ikr = 'ikr'

    def build(self, sens):
        if self is Case.ikr:
            return methods.ikr.build(sens)
        return methods.ap.build(sens)

    @classmethod
    def from_name(cls, name):
        return cls[name]

    @classmethod
    def names(cls):
        return [x.name for x in cls]

    def n_parameters(self):
        return 9

    def sigma0(self, optimiser):
        assert isinstance(optimiser, Optimiser)
        small = (Optimiser.rprop, )

        if self is Case.ikr:
            return 1e-6 if optimiser in small else None
        if self is Case.ap:
            return 0.01 if optimiser in small else None
        raise NotImplementedError


class Optimiser(enum.Enum):
    """
    Enumeration type for optimisation methods.
    """

    cmaes = pints.CMAES
    rprop = pints.IRPropMin

    @classmethod
    def from_name(cls, name):
        return cls[name]

    def log_interval(self, case):
        assert isinstance(case, Case)
        multi = (Optimiser.cmaes, )
        if self in multi:
            return 1    # 10 evals per iter (for IKr)
        return 10

    @classmethod
    def names(cls):
        return [x.name for x in cls]

    def needs_sensitivities(self):
        return self is Optimiser.rprop

