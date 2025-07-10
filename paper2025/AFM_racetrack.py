"""
This script generates the data from Fig. 2 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194)
and uses the mumax+ version v1.1.0.

This simulation can take a significant time to complete (depending on your machine).
"""

import numpy as np
import os
from tqdm import tqdm
import sys

from mumaxplus import World, Grid, Antiferromagnet
from mumaxplus.util.shape import Polygon
from mumaxplus.util.config import twodomain


D = 0.7e-3 # chosen DMI

# magnet parameters (NiO)
# https://doi.org/10.1103/PhysRevB.66.064434
# https://doi.org/10.1103/PhysRevApplied.15.014030
msat = 425e3
a = 4.2e-10
aex = 2.3 * 1.60217663e-21 / a / 2
A_c = -12 * 1.60217663e-21 / a
A_nn = 0
K = 85.7e3 
alpha = 2.1e-4  # https://www.nature.com/articles/nphoton.2010.259

# NiO
# https://doi.org/10.1103/PhysRevMaterials.8.044404
B1 = -3e7 / 2  # divided by 2 because different magnetoelastic energy definition
B2 = -1.7e7 / 2
rho = 6853
C11, C12, C44 = 330e9, 60e9, 110e9

# a bit more damping than 5% damping ratio at 167 GHz, or a shear wave of 8 cell sizes
stiffness_coef = 1e-13  # Rayleigh_damping_stiffness_coefficient(f2, z2)

# --------------------
# calculate some useful quantities beforehand

# simulation grid parameters
nx, ny, nz = 512, 128, 1  # nx very large to avoid back-reflecting waves
cx, cy, cz = 3e-9, 3e-9, 3e-9
length, width, thickness = nx*cx, ny*cy, nz*cz

notch_tip_positions = [300e-9, 450e-9, 600e-9, 750e-9]

v_long = np.sqrt(C11 / rho)
v_shear = np.sqrt(C44 / rho)  # the useful one

wave_length = 0.5e-6

F = 3e8  # quite high, but strongly depends on material
traction_freq = v_shear / wave_length
number_of_periods = 1  # once up and down
traction_period = 1 / traction_freq
traction_time = number_of_periods * traction_period

# let the last excitation definitely get through the whole geometry
extra_time = length / v_shear
total_runtime = traction_time + extra_time

# --------------------
# set up the magnetic simulation

# create a world and a 2D magnet
cellsize = (cx, cy, cz)
world = World(cellsize)
grid = Grid((nx, ny, nz))

geom = Polygon([[notch_tip_positions[0]-15*cx,-cy],[notch_tip_positions[0],6*cy],[notch_tip_positions[0]+15*cx,-cy]])
for notch_tip_pos in notch_tip_positions[1:]:
    geom += Polygon([[notch_tip_pos-15*cx,-cy],[notch_tip_pos,6*cy],[notch_tip_pos+15*cx,-cy]])

magnet = Antiferromagnet(world, grid, geometry=geom.invert())

# magnet parameters without magnetoelastics
magnet.msat = msat
magnet.aex = aex
magnet.alpha = alpha
magnet.afmex_cell = A_c
magnet.afmex_nn = A_nn
magnet.latcon = a
magnet.ku1 = K
magnet.anisU = (1,0,0)
magnet.sub1.dmi_tensor.set_interfacial_dmi(D)
magnet.sub2.dmi_tensor.set_interfacial_dmi(D)

def two_walls(x, y, z):
    if x < notch_tip_positions[1]:
        return twodomain((-1,0,0), (0,0,-1), (1,0,0), notch_tip_positions[0], 2*cx)(x, y, z)
    else:
        return twodomain((1,0,0), (0,0,1), (-1,0,0), notch_tip_positions[2], 2*cx)(x, y, z)

magnet.sub1.magnetization = two_walls
magnet.sub2.magnetization = - magnet.sub1.magnetization.eval()

# relax
print("Relaxing...")
magnet.relax()

# add elastics
magnet.enable_elastodynamics = True
magnet.rho = rho
magnet.C11, magnet.C12, magnet.C44 = C11, C12, C44

# magnetoelastics
magnet.B1 = B1
magnet.B2 = B2

# no adaptive timestepping
world.timesolver.adaptive_timestep = False
world.timesolver.timestep = 10e-15

# --------------------
# damping

magnet.stiffness_damping = stiffness_coef

# -------------------------
# adjust to the new environment for a little bit, instead of "relax"
# but with A LOT of damping

t_adjust_high_damping = 0.180e-9
t_adjust_normal_damping = 0.020e-9

adjust_alpha = 100 * alpha
adjust_coef = 100 * stiffness_coef
adjust_eta = 1e13

print(f"Running magnetoelastics with high damping for {t_adjust_high_damping * 1e9:.3f} ns to let the system adjust...")

old_alpha, old_eta, old_stiffness = magnet.sub1.alpha.eval(), magnet.eta.eval(), magnet.stiffness_damping.eval()

# add a lot of damping (no tanh profile to avoid x100,000 damping)
magnet.alpha = adjust_alpha
magnet.eta = adjust_eta
magnet.stiffness_damping = adjust_coef

world.timesolver.run(t_adjust_high_damping)

# now adjust a little bit further, but with a normal amount of damping
magnet.alpha = old_alpha
magnet.eta = old_eta
magnet.stiffness_damping = old_stiffness

print(f"Running another {t_adjust_normal_damping * 1e9:.3f} ns to adjust with normal damping...")
world.timesolver.run(t_adjust_normal_damping)

# reset time for later
world.timesolver.time = 0

# -------------------------
# add pressure wave via traction

print("Adding traction")

print(f"freq {traction_freq * 1e-9:.5f} GHz, period {1 / traction_freq * 1e9:.5f} ns")

def traction_term(t):
    if t < traction_time:
        return (0, 0, F * np.sin(2*np.pi * traction_freq * t))
    else:
        return (0, 0, 0)

# only left side, not around notches
traction_mask = np.zeros(magnet.boundary_traction.neg_x_side.shape)
traction_mask[:,:,:,0] = 1
magnet.boundary_traction.neg_x_side.add_time_term(term=traction_term, mask=traction_mask)

# --------------------
# run while saving

data_dir = "racetrack_data/"
os.makedirs(data_dir, exist_ok=True)

time_file = data_dir + "time.txt"
state_names = ["mag1", "mag2", "disp", "velo"]
state_qtys = [magnet.sub1.magnetization, magnet.sub2.magnetization,
              magnet.elastic_displacement, magnet.elastic_velocity]
save_name = data_dir + "step_{step}_{name}.npy"

def save_state(step):
    with open(time_file, "a") as file: print(step, world.timesolver.time, file=file)
    for name, qty in zip(state_names, state_qtys):
        np.save(save_name.format(step=step, name=name), qty.eval())

# --------------------
# run, save intermediate states, time and timesteps

runtime_per_step = 0.5e-11
nsteps = int(total_runtime / runtime_per_step)

create_new_data = True
if create_new_data:
    # save initial state
    with open(time_file, "a") as file: print("# step time", file=file)
    save_state(0)

    print(f"running {total_runtime:.5e} s at {runtime_per_step:.5e} s per step, so {nsteps} steps")

    steps = tqdm(range(1, nsteps+1), file=sys.stdout)
    for step in steps:
        world.timesolver._impl.run(runtime_per_step)  # _impl: don't make dt "sensible" all the time
        save_state(step)
