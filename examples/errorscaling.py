"""This script compares the numerical and analytical result of one spin after 10 precessions
in an external magnetic field of 0.1 T with damping. This comparison is done for different time steps
in order to recreate figure 10 of the paper "The design and verification of MuMax3".
However with different algorithms.
https://doi.org/10.1063/1.4899186
"""


import matplotlib.pyplot as plt
import numpy as np
from math import acos, atan, pi, exp, tan, sin, cos, sqrt

from mumaxplus import *
from mumaxplus.util import *


def magnetic_moment_precession(time, initial_magnetization, hfield_z, damping):
    """Return the analytical solution of the LLG equation for a single magnetic
    moment and an applied field along the z direction.
    """
    mx, my, mz = initial_magnetization
    theta0 = acos(mz)
    phi0 = atan(my / mx)
    freq = GAMMALL_DEFAULT * hfield_z / (1 + damping ** 2)
    phi = phi0 + freq * time
    theta = pi - 2 * atan(exp(damping * freq * time) * tan(pi / 2 - theta0 / 2))
    return np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])


def single_system(method, dt):
    """This function simulates a single spin in a magnetic field of 0.1 T without damping.

    Returns the absolute error between the simulation and the exact solution.

    Parameters:
    method -- The used simulation method
    dt     -- The time step
    """
    # --- Setup ---
    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    
    magnetization = (1/np.sqrt(2), 0, 1/np.sqrt(2))
    damping = 0.001
    hfield_z = 0.1  # External field strength
    duration = 2*np.pi/(GAMMALL_DEFAULT * hfield_z) * (1 + damping**2) * 10  # Time of 10 precessions

    magnet = Ferromagnet(world, grid=Grid((1, 1, 1)))
    magnet.enable_demag = False
    magnet.magnetization = magnetization
    magnet.alpha = damping
    magnet.aex = 10e-12
    magnet.msat = 1/MU0
    world.bias_magnetic_field = (0, 0, hfield_z)

    # --- Run the simulation ---
    world.timesolver.set_method(method)
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = dt
    
    world.timesolver.run(duration)
    output = magnet.magnetization.average()

    # --- Compare with exact solution ---
    exact = magnetic_moment_precession(duration, magnetization, hfield_z, damping)
    error = np.linalg.norm(exact - output)

    return error


method_names = ["Heun", "BogackiShampine", "CashKarp", "Fehlberg", "DormandPrince"]

exact_names = {"Heun": "Heun",
               "BogackiShampine": "Bogacki-Shampine",
               "CashKarp": "Cash-Karp",
               "Fehlberg": "Fehlberg",
               "DormandPrince": "Dormand-Prince"
               }

RK_names = {"Heun": "RK12",
            "BogackiShampine": "RK32",
            "CashKarp": "RKCK45",
            "Fehlberg": "RKF45",
            "DormandPrince": "RK45"
            }

exact_order = {"Heun": 2,
               "BogackiShampine": 3,
               "CashKarp": 5,
               "Fehlberg": 5,
               "DormandPrince": 5
               }

N_dens = 30  # Amount of datapoints between two powers of 10

# Lower bounds for the time steps
dts_lower = {"Heun": 0.2e-11,
             "BogackiShampine": 0.4e-11,
             "CashKarp": 0.2e-10,
             "Fehlberg": 1.6e-11,
             "DormandPrince": 1.6e-11
             }

# Upper bounds for the time steps
dts_upper = {"Heun": 0.3e-10,
             "BogackiShampine": 0.5e-10,
             "CashKarp": 0.8e-10,
             "Fehlberg": 0.5e-10,
             "DormandPrince": 0.5e-10
             }

# Time step arrays
dts = {"Heun": np.logspace(np.log10(dts_lower["Heun"]), np.log10(dts_upper["Heun"]), int(N_dens * (np.log10(dts_upper["Heun"]) - np.log10(dts_lower["Heun"])))), 
       "BogackiShampine": np.logspace(np.log10(dts_lower["BogackiShampine"]), np.log10(dts_upper["BogackiShampine"]), int(N_dens * (np.log10(dts_upper["BogackiShampine"]) - np.log10(dts_lower["BogackiShampine"])))),
       "CashKarp": np.logspace(np.log10(dts_lower["CashKarp"]), np.log10(dts_upper["CashKarp"]), int(N_dens * (np.log10(dts_upper["CashKarp"]) - np.log10(dts_lower["CashKarp"])))),
       "Fehlberg": np.logspace(np.log10(dts_lower["Fehlberg"]), np.log10(dts_upper["Fehlberg"]), int(N_dens * (np.log10(dts_upper["Fehlberg"]) - np.log10(dts_lower["Fehlberg"])))),
       "DormandPrince": np.logspace(np.log10(dts_lower["DormandPrince"]), np.log10(dts_upper["DormandPrince"]), int(N_dens * (np.log10(dts_upper["DormandPrince"]) - np.log10(dts_lower["DormandPrince"]))))
       }

# --- Plotting ---
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.9e-12, 1e-10))
plt.ylim((1e-6, 1))
plt.xlabel("time step (s)")
plt.ylabel("absolute error after 10 precession")

plt.plot([], [], color="black", label="Theory")  # Labels for theoretical results
plt.scatter([], [], marker="o", color="black", label="Simulation")  # Labels for simulated results

# --- Simulation Loops ---
orders = {}
for method in method_names:
    error = np.zeros(shape=dts[method].shape)
    for i, dt in enumerate(dts[method]):
        err = single_system(method, dt)
        error[i] = err
    
    # Find the order
    log_dts, log_error = np.log10(dts[method]), np.log10(error)
    order = np.polyfit(log_dts, log_error, 1)[0]
    orders[exact_names[method]] = order

    plt.scatter(dts[method], error, marker="o", zorder=2)

    intercept = np.polyfit(log_dts, log_error - log_dts * exact_order[method], 0)
    plt.plot(np.array([1e-14, 1e-9]), (10**intercept)*np.array([1e-14, 1e-9])**exact_order[method], label=f"{RK_names[method]} {exact_names[method]}")


#print(orders)  # Uncomment if you want to see the estimated orders
plt.legend()
plt.show()
