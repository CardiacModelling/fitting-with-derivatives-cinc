#!/usr/bin/env python3
#
# IKr model with staircase protocol
#
import os

import numpy as np
import pints
import myokit
import myokit.lib.hh

import methods


_model = os.path.join('resources', 'beattie-2017-ikr-hh.mmt')
_proto = os.path.join('resources', 'simplified-staircase.mmt')


class Model(pints.ForwardModel):
    """A forward model that runs IKr simulations with CVODE."""

    def __init__(self):

        # Load a model, and isolate the HH ion current model part
        model = myokit.load_model(os.path.join(methods.DIR, _model))
        parameters = ['ikr.p' + str(1 + i) for i in range(9)]
        hh_model = myokit.lib.hh.HHModel.from_component(
            model.get('ikr'), parameters=parameters)

        # Load a protocol
        protocol = myokit.load_protocol(os.path.join(methods.DIR, _proto))

        # Create a CVODE Simulation
        self.sim = myokit.Simulation(model, protocol)

        # Set solver tolerances
        self.sim.set_tolerance(1e-8, 1e-8)

        # Set the -80mV steady state as the default state
        self.sim.set_default_state(hh_model.steady_state(-80))

    def n_parameters(self):
        return 9

    def simulate(self, parameters, times):

        # Reset to default time and state
        self.sim.reset()

        # Apply parameters
        for i, p in enumerate(parameters):
            self.sim.set_constant('ikr.p' + str(1 + i), p)

        # Run
        tmax = times[-1] + (times[-1] - times[-2])
        try:
            log = self.sim.run(tmax, log_times=times, log=['ikr.IKr'])
            return log['ikr.IKr']
        except myokit.SimulationError:
            print('Error evaluating with parameters: ' + str(parameters))
            return np.nan * times


class ModelWithSensitivities(pints.ForwardModelS1):
    """A forward model that runs IKr simulations with CVODES."""

    def __init__(self):

        # Load a model, and isolate the HH ion current model part
        model = myokit.load_model(os.path.join(methods.DIR, _model))
        parameters = ['ikr.p' + str(1 + i) for i in range(9)]
        hh_model = myokit.lib.hh.HHModel.from_component(
            model.get('ikr'), parameters=parameters)

        # Load a protocol
        protocol = myokit.load_protocol(os.path.join(methods.DIR, _proto))

        # Create a CVODE Simulation
        self.sim = myokit.Simulation(
            model, protocol, sensitivities=(['ikr.IKr'], parameters))

        # Set solver tolerances
        self.sim.set_tolerance(1e-8, 1e-8)

        # Set the -80mV steady state as the default state
        self.sim.set_default_state(hh_model.steady_state(-80))

    def n_parameters(self):
        return 9

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

        # Reset to default time and state
        self.sim.reset()

        # Apply parameters
        for i, p in enumerate(parameters):
            self.sim.set_constant('ikr.p' + str(1 + i), p)

        # Run
        tmax = times[-1] + (times[-1] - times[-2])
        try:
            log, sens = self.sim.run(tmax, log_times=times, log=['ikr.IKr'])
            return log['ikr.IKr'], sens
        except myokit.SimulationError:
            print('Error evaluating with parameters: ' + str(parameters))
            return np.nan * times, np.zeros((len(times), len(parameters)))


class HHModel(pints.ForwardModel):
    """
    A forward model that runs IKr simulations on step protocols, using an
    analytical solving method for Hodgkin-Huxley models.
    """

    def __init__(self, protocol):

        # Load a model, and isolate the HH ion current model part
        model = myokit.load_model(os.path.join(methods.DIR, _model))
        parameters = ['ikr.p' + str(1 + i) for i in range(9)]
        hh_model = myokit.lib.hh.HHModel.from_component(
            model.get('ikr'), parameters=parameters)

        # Create an analytical simulation
        self.sim = myokit.lib.hh.AnalyticalSimulation(hh_model, protocol)

        # Set the -80mV steady state as the default state
        self.sim.set_default_state(hh_model.steady_state(-80))

    def n_parameters(self):
        return 9

    def simulate(self, parameters, times):

        # Reset, apply parameters, and run
        self.sim.reset()
        self.sim.set_parameters(parameters)
        tmax = times[-1] + (times[-1] - times[-2])
        log = self.sim.run(tmax, log_times=times)
        return log['ikr.IKr']


class Boundaries(pints.Boundaries):
    """
    A boundaries class that implements the maximum-rate boundaries used in
    Beattie et al.

    The boundaries operate on untransformed space, but the :meth:`sample`
    method uses a log-transform on all a-type parameters.

    Parameters
    ----------
    g_min
        A cell-specific lower boundary on the conductance.
    """

    # Limits for a-type parameters (untransformed)
    a_min = 1e-7
    a_max = 1e3

    # Limits for b-type parameters
    b_min = 1e-7
    b_max = 0.4

    # Limits for maximum rate coefficients
    km_min = 1.67e-5
    km_max = 1e3

    # Voltages used when determining maximum rate coefficients
    v_low = -120
    v_high = 60

    def __init__(self, g_min=0.1):

        self.g_min = g_min
        self.g_max = 10 * g_min

        # Univariate paramater bounds
        self.lower = np.array([
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.a_min, self.b_min,
            self.g_min,
        ])
        self.upper = np.array([
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.a_max, self.b_max,
            self.g_max,
        ])

    def n_parameters(self):
        return 9

    def check(self, parameters):

        # Check parameter boundaries
        if (np.any(parameters <= self.lower)
                or np.any(parameters >= self.upper)):
            return False

        # Check rate boundaries
        k1m = parameters[0] * np.exp(parameters[1] * self.v_high)
        if k1m <= self.km_min or k1m >= self.km_max:
            return False
        k2m = parameters[2] * np.exp(-parameters[3] * self.v_low)
        if k2m <= self.km_min or k2m >= self.km_max:
            return False
        k3m = parameters[4] * np.exp(parameters[5] * self.v_high)
        if k3m <= self.km_min or k3m >= self.km_max:
            return False
        k4m = parameters[6] * np.exp(-parameters[7] * self.v_low)
        if k4m <= self.km_min or k4m >= self.km_max:
            return False

        # All tests passed!
        return True

    def _sample_partial(self, v):
        """Samples a pair of kinetic parameters"""
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.a_min), np.log(self.a_max)))
            b = np.random.uniform(self.b_min, self.b_max)
            km = a * np.exp(b * v)
            if km > self.km_min and km < self.km_max:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):
        points = np.zeros((n, 9))
        for i in range(n):
            points[i, 0:2] = self._sample_partial(self.v_high)
            points[i, 2:4] = self._sample_partial(-self.v_low)
            points[i, 4:6] = self._sample_partial(self.v_high)
            points[i, 6:8] = self._sample_partial(-self.v_low)
            points[i, 8] = np.random.uniform(self.g_min, self.g_max)
        return points

    def transformed_lower_upper(self):
        """
        Returns a simplified version of these boundaries as a tuple ``(lower,
        upper)``, in transformed space.

        These can be used with scipy optimisers, but don't include the
        boundaries on the rates.
        """
        lower = np.copy(self.lower)
        upper = np.copy(self.upper)
        for i in (0, 2, 4, 6):
            lower[i] = np.log(lower[i])
            upper[i] = np.log(upper[i])
        return lower, upper


class Transformation(pints.ComposedTransformation):
    """
    Log-transformation on all a-type parameters of the IKr model.
    """
    def __init__(self):
        super().__init__(
            pints.LogTransformation(1),
            pints.IdentityTransformation(1),
            pints.LogTransformation(1),
            pints.IdentityTransformation(1),
            pints.LogTransformation(1),
            pints.IdentityTransformation(1),
            pints.LogTransformation(1),
            pints.IdentityTransformation(1),
            pints.IdentityTransformation(1),
        )


def parameters():
    """ Returns a reasonable set of IKr parameters. """
    return np.array([3e-4, 0.07, 3e-5, 0.05, 0.09, 9e-2, 5e-3, 0.03, 0.2])


def build(sensitivities=False):
    """
    Returns a tuple ``(model, problem, error, boundaries, transformation, p)``.
    """
    model = ModelWithSensitivities() if sensitivities else Model()

    # Default parameters
    p = parameters()

    # Noise level
    sigma = 0.025

    # Set up a synthetic data problem and error
    times = np.arange(0, 15400, 0.1)
    values = model.simulate(p, times)
    values += np.random.normal(0, sigma, times.shape)

    # Create a problem and error
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.MeanSquaredError(problem)

    # Create boundaries and a transformation
    boundaries = Boundaries()
    transformation = Transformation()

    return model, problem, error, boundaries, transformation, p

