"""
This script generates the data from Fig. 3 of the paper
'mumax+: extensible GPU-accelerated micromagnetics and beyond' (https://arxiv.org/abs/2411.18194)
and uses the mumax+ version v1.1.1.

This simulation can take a significant time to complete (depending on your machine).
Note that an installation of mumax3 is necessary in order to generate this comparison.
"""


import argparse
import matplotlib.pyplot as plt
import mumaxplus
import numpy as np
import os
import pandas as pd
import re
import requests
import subprocess

from matplotlib.widgets import MultiCursor
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.rc('font', family='serif', size=7)
plt.rc('text', usetex=True)


parser = argparse.ArgumentParser()
parser.add_argument("mpdir", nargs='?', default=r"..\..", help="The mumax+ source directory, needed to recompile between 32- and 64-bit.")
args = parser.parse_args()

MUMAXPLUS_SOURCEDIR: str = args.mpdir
MUMAX3_COMMAND = r"mumax3" # Change to executable path if mumax3 is not in $env:PATH
DATA_DIR = "data"
OUTPUT_DIR = "figures"


def set_mumaxplus_precision(precision: Literal["SINGLE", "DOUBLE"]):
    if precision not in ["SINGLE", "DOUBLE"]: raise ValueError("Precision must be SINGLE or DOUBLE.")
    if precision != mumaxplus.FP_PRECISION:
        env = os.environ.copy()
        env["MUMAXPLUS_FP_PRECISION"] = precision
        subprocess.check_call(["pip", "install", "."], cwd=MUMAXPLUS_SOURCEDIR, env=env)

def run_all(results_dir: str|Path = DATA_DIR):
    results_dir = Path(results_dir).absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {results_dir} ...")
    
    ## OOMMF
    url = "https://www.ctcms.nist.gov/~rdm/std4/Donahue/stdprob4a.odt" # 1nm cell size, OOMMF eXtensible solver
    r = requests.get(url)
    with open(results_dir / "standardproblem4a.odt", 'wb') as outfile:
        outfile.write(r.content)
    
    ## mumax+ and mumax3
    set_mumaxplus_precision("SINGLE")
    for cs in ["2.5nm", "1nm"]: # Do the fastest first
        results_dir_here = results_dir / cs
        subprocess.run(f'python standardproblem4a_{cs}.py "{results_dir_here}"') # mumax+ 32 bit only
        subprocess.run(f'{MUMAX3_COMMAND} -o="{results_dir_here / "standardproblem4a.out"}" standardproblem4a_{cs}.mx3')

def read_ODT(file: str|Path):
    """ Parses the submitted time series at https://www.ctcms.nist.gov/~rdm/std4/Donahue.html """
    file = Path(file)
    lines = []
    with open(file, "r") as inFile:
        for line in inFile:
            parts = line.strip().split(" ")
            parts = [part for part in parts if part]
            lines.append(parts)
    
    columns_str = " ".join(lines[3]).replace("# Columns: ", "").strip()
    columns: list[str] = re.findall(r'\{[^}]+\}|\S+', columns_str)
    columns = [col.strip(r"{}") for col in columns] # .split(":")[-1]
    
    units_str = " ".join(lines[4]).replace("# Units: ", "").strip()
    units: list[str] = re.findall(r'\{[^}]+\}|\S+', units_str)
    units = [unit.strip(r"{}") for unit in units]
    
    data = lines[5:] # First 3 lines are irrelevant for us
    
    df = pd.DataFrame(data, columns=columns)
    df = df.apply(pd.to_numeric, errors='coerce')
    useful_cols = {"Oxs_TimeDriver::Simulation time": "time",
                   "Oxs_TimeDriver::mx": "mx",
                   "Oxs_TimeDriver::my": "my",
                   "Oxs_TimeDriver::mz": "mz",
                   "Oxs_RungeKuttaEvolve:evolver:Total energy": "e_total",
                   "Oxs_UniformExchange::Energy": "e_exchange",
                   "Oxs_FixedZeeman::Energy": "e_zeeman",
                   "Oxs_Demag::Energy": "e_demag"
    }
    
    df_useful = df[[key for key in useful_cols.keys() if key in df.columns]]
    return df_useful.rename(columns=useful_cols)


def main(ERRORLEVEL=0):
    ## LOAD DATA
    try:
        dfs = {
            "OOMMF_1nm": read_ODT(DATA_DIR + "/standardproblem4a.odt"),
            "mumax3_1nm": pd.read_csv(DATA_DIR + "/1nm/standardproblem4a.out/table.txt", sep="\t"),
            "mumax+32_1nm": pd.read_csv(DATA_DIR + "/1nm/standardproblem4a_plus_SINGLE.out", sep="\t")
        }
    except Exception as e:
        if ERRORLEVEL: raise e
        run_all()
        return main(ERRORLEVEL=ERRORLEVEL+1)
    
    for key, df in dfs.items():
        if key.startswith("mumax3"):
            df.rename(columns={"# t (s)": "time", "mx ()": "mx", "my ()": "my", "mz ()": "mz"}, inplace=True)
        elif key.startswith("mumax+"):
            df.rename(columns={"# time": "time"}, inplace=True)
    
    ## PLOTTING
    plot_comps = ["y"]
    fig, axes = plt.subplots(nrows=1, ncols=len(plot_comps), figsize=(2.5*len(plot_comps), 4.8/6.4 * 2.5), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    t_factor, t_unit = 1e9, "n"
    # m_factor, m_unit = 1e3, r"\textperthousand"
    # m_factor, m_unit = 1e2, r"\%"
    m_factor, m_unit = 1e3, r"10^{-3}"
    t_max = 3e-9
    
    tools_names = {"OOMMF": "OOMMF", "mumax3": "mumax$^3$", "mumax+32": r"mumax$^{\!+}$"}
    tools_colors = {"OOMMF": "C6", "mumax3": "C0", "mumax+32": "C1"}
    cs_names = {"1nm": r"$1\!\times\!1\!\times\!1$ nm$^3$", "2.5nm": r"$2.5\!\times\!2.5\!\times\!3$ nm$^3$"}
    cs_linestyles = {"1nm": "-", "2.5nm": "--"}
    lw = .7
    
    base = "OOMMF_1nm"
    df_base = dfs[base]
    t = df_base["time"].to_numpy()
    deviation = lambda df, col: np.interp(x=t, xp=df["time"], fp=df[col]) - df_base[col].to_numpy()
    
    # Difference with baseline, for each magnetisation component
    for i, comp in enumerate(plot_comps):
        ax: plt.Axes = axes[0,i]
        for key, df in dfs.items():
            if key == base:
                ax.plot(t*t_factor, np.zeros_like(t), color="grey", lw=1, ls="-")
                continue
            tool, cs = key.split("_")
            color = tools_colors[tool]
            ls = cs_linestyles[cs]
            ax.plot(t*t_factor, deviation(df, f"m{comp}")*m_factor, color=color, ls=ls, lw=lw)
        
    # Legend entries
    for tool, color in tools_colors.items():
        if tool == base.split("_")[0]: continue
        ax.plot([], [], color=color, label=f"{tools_names[tool]}")
    
    if len(plot_comps) == 1:
        ax.legend(loc="upper right")
    else:
        fig.legend(*ax.get_legend_handles_labels(), ncols=2, loc="upper center")
    t_max = min(df["time"].to_numpy().max() for df in dfs.values())
    ax.set_xlim([0, t_max*t_factor])
    ax.set_xlabel(f"Time ({t_unit}s)")
    ax.set_ylabel(f"$\\Delta \\langle m_{comp} \\rangle$" + f" $/\\,{m_unit}$"*(m_unit != ""))
    in_range = np.where(np.logical_and(t >= 0, t <= t_max))
    ymax = max([np.max(np.abs(deviation(df, f"m{comp}")[in_range])) for df in dfs.values() for comp in plot_comps])*m_factor
    pady = 0.05
    ax.set_ylim([-ymax*(1+pady), ymax*(1+pady)])
    
    multi = MultiCursor(None, tuple(axes.flat), horizOn=False, vertOn=True, color='k', lw=1, ls=":")
    
    fig.tight_layout(h_pad=0)
    fig.subplots_adjust(top=0.98, bottom=0.18)
    for ext in ("pdf", "png", "svg"):
        outfile = Path(f"{OUTPUT_DIR}/comparison_m{comp}_{base}.{ext}").absolute()
        outfile.parent.mkdir(exist_ok=True)
        fig.savefig(outfile, dpi=1200)
    plt.show()


if __name__ == "__main__":
    main()
