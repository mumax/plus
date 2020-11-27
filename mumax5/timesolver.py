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

    def step(self):
        """Make one step of the time solver."""
        self._impl.step()

    def steps(self, nsteps):
        """Make n steps of the time solver."""
        for i in range(nsteps):
            self.step()

    def run(self, duration):
        """Run the solver for a given duration.

        Parameters
        ----------
        duration : float
            Duration in seconds.
        """
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
        # check if time points are OK
        output = TimeSolverOutput(quantity_dict)
        for tp in timepoints:
            duration = tp - self.time
            self.run(duration)
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
