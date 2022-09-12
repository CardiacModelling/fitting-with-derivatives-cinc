#!/usr/bin/env python3
#
# Kernik hIPSC model with vcp_opt protocol (naive version for benchmarking)
#
import os

import myokit
import pints

import numpy as np

import methods


_model = os.path.join('resources', 'kernik-2019.mmt')
_proto = os.path.join('resources', 'vcp_opt.txt')


def _prepare_model(model):
    """ Clamps concentrations, applies VC, returns conductance params. """
    model.validate()

    # Check currents and conductance/permeabilities
    cs = ['Na', 'CaL', 'Kr', 'Ks', 'to', 'NaCa', 'K1', 'NaK', 'f', 'CaT']
    conductances = []
    for suffix in cs:
        var = model.label('g_' + suffix)
        if var is None:
            var = model.label('P_' + suffix)
        conductances.append(var)
        var.clamp()

    # Clamp concentrations
    con = {
        'Na_i': 10,
        'Na_o': 137,
        'K_i': 130,
        'K_o': 5.4,
        'Ca_o': 2,
        #'Ca_i': 1e-5,
    }
    for var, value in con.items():
        var = model.labelx(var).clamp(value)

    # Apply voltage clamp
    pace = model.binding('pace')
    if pace is not None:
        pace.set_binding(None)

    vm = model.labelx('membrane_potential')
    vm.clamp(0)
    vm.set_binding('pace')

    return conductances


def _vcp_opt_steps():
    """ Returns a protocol with the steps (not the ramps) for ``vcp_opt``. """
    p = np.loadtxt(os.path.join(methods.DIR, _proto)).flatten()
    protocol = myokit.Protocol()
    for i in range(0, len(p), 2):
        protocol.add_step(p[i], p[i + 1])
    return protocol


def _vcp_opt_ramps(model):
    """ Adds the ``vcp_opt`` ramps to a ``myokit.Model``. """

    # Ramps: t0, dt, dv
    ramps = (
        (1971, 103.7, -32.110413609918844),
        (2757.7, 101.9, 65.36745364287121),
        (3359.6, 272.1, -161.43947608426976),
        (4234.5, 52.2, -4.17731209749318),
        (6429.2, 729.4, 0.6554508694203413),
        (8155.2, 894.9, -30.687646597436324),
    )

    # Unbind pacing variable
    p = model.binding('pace')
    p.set_binding(None)

    # Introduce new paced varaible
    q = p.parent().add_variable_allow_renaming('pace_new')
    q.set_rhs(0)
    q.set_binding('pace')

    # Replace original pacing variable rhs with ref to new variable
    # and/or ramps.
    tn = model.time().qname()

    # Create list of conditions and values for piecewise
    args = []
    for t0, dt, dv in ramps:
        args.append(f'{tn} >= {t0} and {tn} < {t0 + dt}')
        args.append(f'({tn} - {t0}) * {dv / dt}')
    args.append('0')
    p.set_rhs(q.name() + ' + piecewise(' + ', '.join(args) + ')')


class Model(pints.ForwardModel):
    """
    A :class`pints.ForwardModel` representing a stem cell under voltage-clamp
    conditions.

    Note: This is a primitive implementation for benchmarking purposes. It
    makes unrealistic assumptions (e.g. known initial conditions and idealised
    voltage clamp).
    """
    def __init__(self):

        # Load model
        self._model = myokit.load_model(os.path.join(methods.DIR, _model))
        self._parameters = _prepare_model(self._model)
        self._current = self._model.labelx('cellular_current')

        # Load protocol
        _vcp_opt_ramps(self._model)
        self._protocol = _vcp_opt_steps()

        # Original parameter values
        self._porgs = [self._model.get(x).eval() for x in self._parameters]

        # Simulation
        self.sim = myokit.Simulation(self._model, self._protocol)

        # Set solver tolerances
        self.sim.set_tolerance(1e-8, 1e-8)

    def n_parameters(self):
        return 10

    def simulate(self, parameters, times):

        # Reset to default time and state
        self.sim.reset()

        # Apply parameters
        for var, x, y in zip(self._parameters, self._porgs, parameters):
            self.sim.set_constant(var, x * y)

        # Run
        tmax = times[-1] + (times[-1] - times[-2])
        try:
            log = self.sim.run(tmax, log_times=times, log=[self._current])
            return log[self._current]
        except myokit.SimulationError:
            print('Error evaluating with parameters: ' + str(parameters))
            return np.nan * times


class ModelWithSensitivities(pints.ForwardModelS1):
    """A forward model that runs AP simulations with CVODES."""

    def __init__(self):

        # Load model
        self._model = myokit.load_model(os.path.join(methods.DIR, _model))
        self._parameters = _prepare_model(self._model)
        self._current = self._model.labelx('cellular_current')

        # Load protocol
        _vcp_opt_ramps(self._model)
        self._protocol = _vcp_opt_steps()

        # Original parameter values
        self._porgs = np.array(
            [self._model.get(x).eval() for x in self._parameters])

        # Simulation
        self.sim = myokit.Simulation(
            self._model,
            self._protocol,
            sensitivities=([self._current], self._parameters))

        # Set solver tolerances
        self.sim.set_tolerance(1e-8, 1e-8)

    def n_parameters(self):
        return 10

    def simulate(self, parameters, times):
        return self.simulateS1(parameters, times)[0]

    def simulateS1(self, parameters, times):

        # Returns
        # -------
        # y
        #    The simulated values, as a sequence of ``n_times`` values, or
        #    a NumPy array of shape ``(n_times, n_outputs)``.
        # y'
        #    The corresponding derivatives, as a NumPy array of shape
        #    ``(n_times, n_parameters)`` or an array of shape
        #    ``(n_times, n_outputs, n_parameters)``.

        # Note:
        #  We calculate dy/dp where p is the adjusted parameter p=f*p_org
        #   and f is the scaling factor
        #  We want dy/df = dy/dp * dp/df
        #   dp/df = p_org, so we need to multiply the output by p_org
        #

        # Reset to default time and state
        self.sim.reset()

        # Apply parameters
        for var, x, y in zip(self._parameters, self._porgs, parameters):
            self.sim.set_constant(var, x * y)

        # Run
        tmax = times[-1] + (times[-1] - times[-2])
        try:
            d, s = self.sim.run(tmax, log_times=times, log=[self._current])
            d = d.npview()[self._current]
            s = np.array(s) * self._porgs[None, None, :]
            return d, s
        except myokit.SimulationError:
            print('Error evaluating with parameters: ' + str(parameters))
            return np.nan * times, np.zeros((len(times), len(parameters)))


class APBoundaries(pints.RectangularBoundaries):
    """
    Rectangular boundaries that sample in a log transformed space.
    """
    def sample(self, n=1):
        return np.exp(np.random.uniform(
            np.log(self._lower),
            np.log(self._upper),
            size=(n, self._n_parameters)
        ))


def parameters():
    """ Returns the default set of AP parameters. """
    return np.ones(10)


def build(sensitivities=False):
    """
    Returns a tuple ``(model, problem, error, boundaries, transformation, p)``.
    """
    model = ModelWithSensitivities() if sensitivities else Model()

    # Default parameters
    p = parameters()

    # Noise level
    sigma = 0.1

    # Set up a synthetic data problem and error
    times = np.arange(0, 9050, 0.1)
    values = model.simulate(p, times)
    values += np.random.normal(0, sigma, times.shape)

    # Create a problem and error
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.MeanSquaredError(problem)

    # Create boundaries and a transformation
    boundaries = APBoundaries(np.ones(10) * 1e-3, np.ones(10) * 1e3)
    transformation = pints.LogTransformation(10)

    return model, problem, error, boundaries, transformation, p

