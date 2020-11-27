"""Classes for solving differential equations in the time domain."""

import _mumax5cpp as _cpp


class TimeSolverOutput:
    """Collect values of a list of quantities on specified timepoints.

    Parameters
    ----------
    quantity_dict : dict
        Quantities to collect.
    """

    def __init__(self, quantity_dict):
        self._quantities = quantity_dict
        self._data = {"time": []}
        for key in self._quantities.keys():
            self._data[key] = []

    def write_line(self, time):
        """Compute all the specified quantities for the current state."""
        self._data["time"].append(time)
        for key, func in self._quantities.items():
            self._data[key].append(func())

    def __getitem__(self, key):
        """Return the computed values of a quantity."""
        return self._data[key]


class TimeSolver:
    """Solve a set of differential equations in the time domain.

    Parameters
    ----------
    variable : mumax5.Variable
        Independent variable.
    rhs : mumax5.FieldQuantity
        The right-hand side of a differential equation.
    """

    def __init__(self, variable, rhs):
        self._impl = _cpp.TimeSolver(variable._impl, rhs._impl)

    @classmethod
    def _from_impl(cls, impl):
        solver = cls.__new__(cls)
        solver._impl = impl
        return solver

    def _assure_sensible_timestep(self):
        """Assure a sensible timestep.

        If things in the world have been changed, than it could be that the current
        timestep of the solver is way to big. Calling this method makes sure that
        the timestep is sensibly small.
        """
        if self.adaptive_timestep:
            if self.timestep == 0.0 or self.timestep > self._impl.sensible_timestep:
                self.timestep = self._impl.sensible_timestep

    def steps(self, nsteps):
        """Make n steps with the time solver."""
        # Make sure we start stepping with a sensible timestep
        self._assure_sensible_timestep()
        self._impl.steps(nsteps)

    def run(self, duration):
        """Run the solver for a given duration.

        Parameters
        ----------
        duration : float
            Duration in seconds.
        """
        self._assure_sensible_timestep()
        self._impl.run(duration)

    def solve(self, timepoints, quantity_dict):
        """Solve the differential equation.

        The functions collects values of a list of specified quantities
        on specified timepoints.

        Parameters
        ----------
        timepoints : iterable[float]
            Specified timepoints.
        quantity_dict : dict
            Specified quantities to collect.
        """
        # TODO:check if time points are OK
        self._assure_sensible_timestep()
        output = TimeSolverOutput(quantity_dict)

        for tp in timepoints:
            # we only need to assure a sensible timestep at the beginning,
            # hence we use here self._impl.run instead of self.run
            duration = tp - self.time
            self._impl.run(duration)

            output.write_line(tp)
        return output

    @property
    def timestep(self):
        """Return the timestep value."""
        return self._impl.timestep

    @timestep.setter
    def timestep(self, timestep):
        self._impl.timestep = timestep

    @property
    def adaptive_timestep(self):
        """Return the adaptive_timestep value.

        True if an adaptive time step is used, False otherwise.
        To enable an adaptive time step, set to True, to disable set to False.
        """
        return self._impl.adaptive_timestep

    @adaptive_timestep.setter
    def adaptive_timestep(self, adaptive):
        self._impl.timestep = adaptive

    @property
    def time(self):
        """Return the time value."""
        return self._impl.time
