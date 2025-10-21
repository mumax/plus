# Paper directory
The current directory contains the scripts to generate the data from the paper
> mumax+: extensible GPU-accelerated micromagnetics and beyond

https://arxiv.org/abs/2411.18194

All demonstrations in the paper were simulated using version [v1.1.0](https://github.com/mumax/plus/tree/v1.1.0) of the code. All dependencies which are needed to generate the data from the paper, are included in the main installation of mumax⁺ and can be found in the main [README](https://github.com/mumax/plus/tree/master?tab=readme-ov-file#dependencies) of the repository.

A short summary of the available scripts for each section in the paper is given below. Each Python file also contains a short docstring with additional information on the simulation.

## Methods

* The data in Fig. 1 can be generated with [anti_AFM.py](https://github.com/mumax/plus/tree/paper2025/paper2025/antiAFM.py)
* The benchmark data in Fig. 2 can be generated with [bench.py](https://github.com/mumax/plus/tree/paper2025/paper2025/bench.py). Note that you will need an installation of mumax³ if you want to generate the complete plot.
* The data in Fig. 3 can be generated with the scripts in the [standardproblem4](https://github.com/mumax/plus/tree/paper2025/paper2025/standardproblem4) directory. Use [standardproblem4a_1nm.py](https://github.com/mumax/plusmumax/plus/tree/paper2025/paper2025/standardproblem4/standardproblem4a_1nm.py) for the mumax⁺ data and use [standardproblem4a_1nm.mx3](https://github.com/mumaxmumax/plus/tree/paper2025/paper2025//standardproblem4/standardproblem4a_1nm.mx3) for the mumax³ data. The [compare.py](https://github.com/mumaxmumax/plus/tree/paper2025/paper2025/standardproblem4/compare.py) script can then be used to compare the results with OOMMF. Note that you need an installation of mumax³ if you want to generate the complete plot.

## Demonstrations

* The data from Fig. 4 is generated with [AFM_mobility.py](https://github.com/mumax/plus/tree/paper2025/paper2025/AFM_mobility.py).
* The data from Fig. 5 is generated with [AFM_racetrack.py](https://github.com/mumax/plus/tree/paper2025/paper2025/AFM_racetrack.py).
* The data from Fig. 6 is generated with [NcAfm_NV.py](https://github.com/mumax/plus/tree/paper2025/paper2025/NcAfm_NV.py).