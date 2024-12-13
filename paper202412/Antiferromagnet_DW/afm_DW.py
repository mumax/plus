"""
This is the script used to generate the antiferromagnetic dispertion
relation plots in the paper 
"mumax⁺: extensible GPU-accelerated micromagnetics and beyond"
https://arxiv.org/abs/2411.18194

This script compares results of mumaxplus to those from the paper of
Sánchez-Tejerina et al. (https://doi.org/10.1103/PhysRevB.101.014433)
They describe an analytical model of antiferromagnetic domain wall
motion driven by a current and give theoretical results for it's
width and speed.

The numerical material parameters used here can be found
in the mentioned article.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os.path

from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util import *
import matplotlib

font = {'family' : 'serif',
        'size'   : 7}

matplotlib.rc('font', **font)
plt.rc('text', usetex=True)


cx, cy, cz = 2e-9, 2e-9, 2e-9  # Cellsize
nx, ny, nz = 1, 100, 200  # Number of cells
dw = 4  # Width of the domain wall in number of cells
z = np.linspace(-ny*cy, ny*cy, nz)  # Sample range when fitting

def compute_domain_wall_width(magnet, p1=np.pi, p2=0):
    """Computes the domain wall width"""
    ku = magnet.sub1.ku1.uniform_value
    aex = magnet.sub1.aex.uniform_value
    afmex_nn = magnet.afmex_nn.uniform_value

    phi2 = np.min(np.arccos(magnet.sub1.magnetization()[2, :, ny//2, :]))
    phi1 = np.max(np.arccos(magnet.sub2.magnetization()[2, :, ny//2, :]))
    phi2 = p2
    phi1 = p1
    fac = (2 - np.cos(phi1 - phi2)) / 3
    hexch = 4 * magnet.afmex_cell.uniform_value / (0.35e-9)**2
    denom = hexch * (np.cos(phi1 - phi2) + 1)

    return np.sqrt((2*aex - afmex_nn*fac) / (2*ku - denom))

def compute_domain_wall_speed(magnet, J, p1, p2):
    """Computes the domain wall speed"""
    pol = magnet.sub1.pol.uniform_value
    #J = magnet.sub1.jcur.uniform_value[-1]
    FL = magnet.sub1.free_layer_thickness.uniform_value
    Ms = magnet.sub1.msat.uniform_value
    alpha = magnet.sub1.alpha.uniform_value

    Hsh = -HBAR * pol * J / (2 * QE * FL)
    L = 2*Ms / GAMMALL

    phi2 = np.min(np.arccos(magnet.sub1.magnetization()[2, :, ny//2, :]))
    phi1 = np.max(np.arccos(magnet.sub2.magnetization()[2, :, ny//2, :]))

    return np.pi * fit_domain_wall(magnet)[-1] * Hsh / (alpha * 2 * L) * (np.cos(p1) - np.cos(p2))


def DW_profile(x, position, width):
    """Walker ansatz to describe domain wall profile"""
    return np.cos(2 * np.arctan(np.exp(-(x - position) / width)))

def fit_domain_wall(magnet):
    """Fit Walker ansatz to domain wall profile"""
    mz = magnet.sub1.magnetization()[0,]
    profile = mz[:, int(ny/2), 0]  # middle row of the grid
    popt, pcov = curve_fit(DW_profile, z, profile, p0=(1e-9, 5e-9))
    return popt

def get_domain_wall_speed(magnet):
    """Find stationary value of the velocity"""
    t = 4e-11 / (j + 1)
    #t = 2e-11
    magnet.world.timesolver.run(t/2)
    q1 = fit_domain_wall(magnet)[0]
    magnet.world.timesolver.run(t/2)
    q2 = fit_domain_wall(magnet)[0]

    phi2 = np.min(np.arccos(magnet.sub1.magnetization()[2, :, ny//2, :]))
    phi1 = np.max(np.arccos(magnet.sub2.magnetization()[2, :, ny//2, :]))

    pos = int((q2 - q2 % cz)/cz)
    return 2 * np.abs(q1-q2) / t, phi1, phi2


twodomain_m1 = twodomain((-1,0,0), (0,0,1), (1,0,0), nz*cz/2, dw*cz)  # Domain for first sublatice
twodomain_m2 = twodomain((1,0,0), (0,0,-1), (-1,0,0), nz*cz/2, dw*cz)  # Domain for second sublatice

# Switch x and z to get the domain wall perpendicular to the z-axis
def rotated_twodomain_m1(x, y ,z):
    return twodomain_m1(z, y, x)
def rotated_twodomain_m2(x, y, z):
    return twodomain_m2(z, y, x)
    
def initialize(magnet):
    """Create a two-domain state"""
    magnet.jcur = (0, 0, 0)
    nz2 = nz // 2
    dw2 = dw // 2
    m = np.zeros(magnet.sub1.magnetization.shape)
    m[0,         0:nz2 - dw2, :, :] = -1
    m[2, nz2 - dw2:nz2 + dw2, :, :] = 1  # Domain wall has a width of 4 nm.
    m[0, nz2 + dw2:         , :, :] = 1
    
    magnet.sub1.magnetization = m
    magnet.sub2.magnetization = -m
    world.timesolver.run(1e-11)  # instead of minimize for better performance

world = World(cellsize=(cx, cy, cz))
magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 0.4e6
magnet.alpha = 0.1
magnet.anisU = (1, 0, 0)
magnet.aex = 10e-12
magnet.afmex_cell = -25e-12
magnet.afmex_nn = -5e-12
magnet.sub1.dmi_tensor.set_interfacial_dmi(0.11e-3)
magnet.sub2.dmi_tensor.set_interfacial_dmi(0.11e-3)

initialize(magnet)

static = True
dynamic = True

if static:
    Krange = np.linspace(40, 100, 7)
    DWdata = []
    Aex = [10, 20, 30]
    fig = plt.figure(figsize=(2.5, 4.8/6.4 * 2.5))
    for i, a in enumerate(Aex):
        magnet.aex = a * 1e-12
        DWs = []
        if os.path.isfile("DWdata_width.npy"):
            DWs = np.load("DWdata_width.npy")[i]
        else:
            for K in Krange:
                magnet.ku1 = K * 1e3
                initialize(magnet)
                DWs.append(fit_domain_wall(magnet)[-1]*1e9)

        x = np.linspace(35, 105, 100)
        lab = "Analytical model" if a == 10 else ""
        plt.plot(x, np.sqrt((2*a*1e-12 - (-5e-12)) / (2*x*1e3))*1e9, 'k--', label=lab, lw=0.5)
        plt.plot(Krange, DWs, '.', label=r"$A_{{11}}$ = {} pJ/m".format(a))
        DWdata.append(DWs)

    plt.legend(fontsize="5")
    plt.xlim(np.min(x), np.max(x))
    plt.xlabel(r"Anisotropy constant $K_{u1}$ (kJ/m$^3$)")
    plt.ylabel(r"Domain wall width (nm)")
    plt.tight_layout()
    plt.savefig("dww.pdf", dpi=1200)

    np.save("DWdata_width", DWdata)

if dynamic:
    DWdata = []
    DWtheo = []
    fig = plt.figure(figsize=(2.5, 4.8/6.4 * 2.5))
    jrange = np.linspace(0, 3.5, 9)
    arange = np.linspace(10, 40, 4)
    lab = True
    magnet.ku1 = 64e3
    magnet.aex = 2e-12
    magnet.afmex_cell = -2e-12
    magnet.pol = 0.044
    magnet.fixed_layer = (0, 1, 0)
    magnet.Lambda = 1
    magnet.free_layer_thickness = 2e-9
    for i, a in enumerate(arange):
        magnet.afmex_nn = -a * 1e-12
        speeds = []
        theory = []
        phi1s, phi2s = [], []
        if os.path.isfile("DWdata_speed.npz"):
            data = np.load("DWdata_speed.npz")
            speeds = data["array1"][i]
            theory = data["array2"][i]
        else:
            for j in jrange:
                initialize(magnet)
                magnet.jcur = (0, 0, j * 1e12)
                s, p1, p2 = get_domain_wall_speed(magnet)
                speeds.append(s)
                phi1s.append(p1)
                phi2s.append(p2)
                theory.append(compute_domain_wall_speed(magnet, j*1e12, p1, p2))

        plt.plot(jrange, [t * 1e-3 for t in theory], 'k--', label=lab*"Analytical model", lw=0.5)
        plt.plot(jrange[1:8], [s * 1e-3 for s in speeds[1:8]], '.', label=r"$A_{{12}}$ = -{} pJ/m".format(int(a)))
        lab = False
        DWdata.append(speeds)
        DWtheo.append(theory)
        
    plt.xlim(0, 3.5)
    plt.xlabel(r"Current density $J$ (TA/m$^2$)")
    plt.ylabel(r"Velocity (km/s)")
    plt.legend(fontsize="5")
    plt.tight_layout()
    plt.savefig("dwspeed.pdf", dpi=1200)
    np.savez('DWdata_speed.npz', array1=DWdata, array2=DWtheo)