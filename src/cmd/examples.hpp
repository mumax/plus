/** Standard problem 4 example. */
void standard_problem4();
/** Skyrmion excitation example. */
void skyrmion_excitation();
/** Ferromagnetic spinwave dispersion relation.
* 
* Disregarding dipole - dipole interactions, the dispersion relation $f(k)$ in a ferromagnetic wire is given by
* $$  f(k) = \frac{ \gamma }{2\pi}\left[\frac{ 2A }{M_s} k ^ 2 + B \right] $$
* with $A$ the exchange stiffness, $M_s$ the saturation magnetization, and $B$ an externally applied field.In this section,
* we will try to reproduce this dispersion relation numerically using mumax3.
* 
* The mumax3 script below simulates a uniformly magnetized nanowire with an applied field perpendicular to the wire.
* Spin waves are excited by a sinc pulse with a maximum frequency of 20GHz at the center of the simulation box.Note that the
* demagnetization is disabled in this simulation.
*/
void spinwave_dispersion();
