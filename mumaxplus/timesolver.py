"""Classes for solving differential equations in the time domain."""

from typing import Callable
import os as _os
from tqdm import tqdm as _tqdm

class TimeSolverOutput:
    """Collect values of a list of quantities on specified timepoints.

    Parameters
    ----------
    quantity_dict : dict
        Quantities to collect.
    file_name : str, optional
        Optional name of an output file, in which the data is also written as
        tab-separated values.
    """

    def __init__(self, quantity_dict, file_name=None):
        self._quantities = quantity_dict
        self._data = {"time": []}
        for key in self._quantities.keys():
            self._data[key] = []

        self._keys = list(self._data.keys())  # keep list of keys to maintain order
        self._file_name = file_name
        if self._file_name is not None:
            if directory := _os.path.dirname(self._file_name):
                _os.makedirs(directory, exist_ok=True)
            with open(self._file_name, 'w') as file:  # make new file
                print("# " + "\t".join(self._keys), file=file)

    def write_line(self, time):
        """Compute all the specified quantities for the current state."""
        self._data["time"].append(time)
        for key, func in self._quantities.items():
            self._data[key].append(func())

        # write all latest data to a new line in file
        if self._file_name is not None:
            with open(self._file_name, 'a') as file:
                print(*[self._data[key][-1] for key in self._keys], sep="\t", file=file)

    def __getitem__(self, key):
        """Return the computed values of a quantity."""
        return self._data[key]


class TimeSolver:
    """Evolve the world in time.

    Each world has already its own TimeSolver. This TimeSolver can be accessed through
    the world.timesolver property.

    TimeSolvers should not be initialized by the end user.
    """

    def __init__(self, impl):
        self._impl = impl

    def _assure_sensible_timestep(self):
        """Assure a sensible timestep.

        If things in the world have been changed, than it could be that the current
        timestep of the solver is way to big. Calling this method makes sure that
        the timestep is sensibly small.
        """
        if self.adaptive_timestep:
            if self.timestep == 0.0 or self.timestep > self._impl.sensible_timestep:
                self.timestep = self._impl.sensible_timestep

    def set_method(self, method_name):
        """Set the Runga Kutta method used by the time solver.

        Implemented methods are:
          'Heun'
          'BogackiShampine'
          'CashKarp'
          'Fehlberg'
          'DormandPrince'

        The default and recommended Runge Kutta method is 'Fehlberg'.
        """
        self._impl.set_method(method_name)

    def steps(self, nsteps):
        """Make n steps with the time solver."""
        # Make sure we start stepping with a sensible timestep
        self._assure_sensible_timestep()
        self._impl.steps(nsteps)

    def run_while(self, condition: Callable[[], bool]):
        """Run the solver while the evaluation of the condition is True.

        Parameters
        ----------
        condition : Callable[[], bool]
            Callable condition returning a boolean.
        """
        self._assure_sensible_timestep()
        self._impl.run_while(condition)

    def run(self, duration):
        """Run the solver for a given duration.

        Parameters
        ----------
        duration : float
            Duration in seconds.
        """
        self._assure_sensible_timestep()
        self._impl.run(duration)

    def solve(self, timepoints, quantity_dict, file_name=None, tqdm=False) -> "TimeSolverOutput":
        """Solve the differential equation.

        The functions collects values of a list of specified quantities
        on specified timepoints.

        Parameters
        ----------
        timepoints : iterable[float]
            Specified timepoints.
        quantity_dict : dict
            Specified quantities to collect.
        file_name : str, optional
            Optional name of an output file, in which the data is also written
            as tab-separated values during the simulation.
        tqdm : bool (default=False)
            Prints tqdm progress bar if set to True.

        Returns
        -------
        output : TimeSolverOutput
            Collected values of specified quantities at specified timepoints.
        """
        # check if time points are increasing and lie in the future
        assert all(i1 <= i2 for i1, i2 in zip(timepoints, timepoints[1:])), "The list of timepoints should be increasing."
        assert self.time <= timepoints[0], "The list of timepoints should lie in the future."

        self._assure_sensible_timestep()
        output = TimeSolverOutput(quantity_dict, file_name)

        if tqdm: timepoints = _tqdm(timepoints)
        for tp in timepoints:
            # we only need to assure a sensible timestep at the beginning,
            # hence we use here self._impl.run instead of self.run
            duration = tp - self.time
            self._impl.run(duration)

            output.write_line(self.time)
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
        self._impl.adaptive_timestep = adaptive

    @property
    def time(self):
        """Return the time value."""
        return self._impl.time

    @time.setter
    def time(self, time):
        self._impl.time = time

    @property
    def max_error(self):
        """Return the maximum error per step the solver can tollerate.
        
        The default value is 1e-5.

        See Also
        --------
        headroom, lower_bound, sensible_factor, sensible_timestep_default, upper_bound
        """

        return self._impl.max_error

    @max_error.setter
    def max_error(self, error):
        assert error > 0, "The maximum error should be bigger than 0."
        self._impl.max_error = error
    
    @property
    def headroom(self):
        """Return the solver headroom.
        
        The default value is 0.8.

        See Also
        --------
        lower_bound, max_error, sensible_factor, sensible_timestep_default, upper_bound
        """
        return self._impl.headroom

    @headroom.setter
    def headroom(self, headr):
        assert headr > 0, "The headroom should be bigger than 0."
        self._impl.headroom = headr
    
    @property
    def lower_bound(self):
        """Return the lower bound which is used to cap the scaling of the time step
        from below.
        
        The default value is 0.5.

        See Also
        --------
        headroom, max_error, sensible_factor, sensible_timestep_default, upper_bound
        """
        return self._impl.lower_bound

    @lower_bound.setter
    def lower_bound(self, lower):
        assert lower > 0, "The lower bound should be bigger than 0."
        self._impl.lower_bound = lower
    
    @property
    def upper_bound(self):
        """Return the upper bound which is used to cap the scaling of the time step
        from the top.
        
        The default value is 2.0.

        See Also
        --------
        headroom, lower_bound, max_error, sensible_factor, sensible_timestep_default
        """
        return self._impl.upper_bound

    @upper_bound.setter
    def upper_bound(self, upper):
        assert upper > 0, "The upper bound should be bigger than 0."
        self._impl.upper_bound = upper
    
    @property
    def sensible_factor(self):
        """Return the sensible time step factor which is used as a scaling factor
        when determining a sensible timestep.
        
        The default value is 0.01.

        See Also
        --------
        headroom, lower_bound, max_error, sensible_timestep_default, upper_bound
        """
        return self._impl.sensible_factor

    @sensible_factor.setter
    def sensible_factor(self, fact):
        assert fact > 0, "The sensible factor should be bigger than 0."
        self._impl.sensible_factor = fact

    @property
    def sensible_timestep_default(self):
        """Return the time step which is used if no sensible time step
        can be calculated (e.g. when the total torque is zero).

        The default value is 1e-14 s.

        See Also
        --------
        headroom, lower_bound, max_error, upper_bound
        """
        return self._impl.sensible_timestep_default

    @sensible_timestep_default.setter
    def sensible_timestep_default(self, dt):
        self._impl.sensible_timestep_default = dt