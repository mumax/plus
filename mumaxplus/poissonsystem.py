"""PoissonSolver implementation."""


class PoissonSystem:
    """Poisson System which can be solver for the electrostatic potential."""
    def __init__(self, impl):
        self._impl = impl

    def solve(self):
        """Solve the Poisson equation."""
        return self._impl.solve()

    def set_method(self, method_name) -> str:
        """Set the solver method.

        The implemented methods are:

        * jacobi
        * conjugategradient
        * minimalresidual
        * steepestdescent
        
        """
        return self._impl.solver.set_method(method_name)

    @property
    def max_iter(self) -> int:
        """Maximum of iterations for the solver (no maximum if < 0)."""
        return self._impl.solver.max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._impl.solver.max_iter = max_iter

    @property
    def tol(self) -> float:
        """Error tollerance on the max norm of the residual."""
        return self._impl.solver.tol

    @tol.setter
    def tol(self, tol):
        self._impl.solver.tol = tol

    def max_norm_residual(self) -> float:
        """Return the maximum norm of the residual."""
        return self._impl.solver.max_norm_residual()

    # ----------------------------------------------------------------------------------
    # The hidden methods below should only be used to test and debug the Poisson solver

    @property
    def _solver(self):
        return self._impl.solver

    def _init(self):
        """Initializes/resets the poisson solver."""
        self._impl.init()
