import os
os.environ["GPU_ABORT"] = "0"  # make GPU out-of-memory a catchable RuntimeError instead of aborting

import argparse
import json
import subprocess
import sys
import tempfile
import time

import matplotlib.pyplot as plt


def _smi(*args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True).stdout.strip()


def list_gpus():
    """[(index, name), ...] as reported by nvidia-smi."""
    gpus = []
    for line in _smi("--query-gpu=index,name", "--format=csv,noheader").splitlines():
        idx, name = [x.strip() for x in line.split(",", 1)]
        gpus.append((idx, name))
    return gpus


def get_used_gpu_name():
    """Name of the GPU this process runs on. Call after CUDA has been initialized."""
    # uuid -> (index, name)
    gpus = {}
    for line in _smi("--query-gpu=uuid,index,name", "--format=csv,noheader").splitlines():
        uuid, idx, name = [x.strip() for x in line.split(",", 2)]
        gpus[uuid] = (idx, name)

    # 1) Best: which GPU does nvidia-smi say our PID is running on?
    pid = str(os.getpid())
    used = []
    for line in _smi("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader").splitlines():
        p, uuid = [x.strip() for x in line.split(",", 1)]
        if p == pid and uuid in gpus:
            used.append(gpus[uuid][1])
    if used:
        return ", ".join(used)

    # 2) Fallback (e.g. PID namespaces in containers): use CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible:
        for uuid, (idx, name) in gpus.items():
            if visible == idx or uuid.startswith(visible):  # index or (partial) UUID
                return name

    # 3) No selection made: CUDA's default device. Only a guess if multiple GPUs.
    names = [n for _, n in gpus.values()]
    return names[0] if len(names) == 1 else "unknown (default GPU of " + ", ".join(names) + ")"


def simple_bench(grid, nsteps=100):
    """Returns the walltime of a simple simulation using the specified grid
    and number of steps"""
    from mumaxplus import Ferromagnet, World  # imported lazily so the parent never initializes CUDA

    world = World((4e-9, 4e-9, 4e-9))

    magnet = Ferromagnet(world, grid)
    magnet.msat = 800e3
    magnet.aex = 13e-12
    magnet.alpha = 0.5

    world.timesolver.set_method('Fehlberg')
    world.timesolver.timestep = 1e-13
    world.timesolver.adaptive_timestep = False

    world.timesolver.steps(10)  # warm up

    start = time.time()
    world.timesolver.steps(nsteps)
    stop = time.time()

    return stop - start


def run_benchmark(nsteps=100):
    """Benchmark on whatever GPU this process uses.
    Returns (gpu_name, [(ncells, walltime, throughput), ...])."""
    from mumaxplus import Grid

    simple_bench(Grid((4, 4, 1)), 1)  # tiny run to initialize CUDA so the GPU can be identified
    gpu_name = get_used_gpu_name()

    print("\nGPU: ", gpu_name)
    print("{:>10} {:>10} {:>12}".format("ncells", "walltime", "throughput"))

    results = []
    p = 2
    while True:
        try:
            grid = Grid((2 ** p, 2 ** p, 1))
            walltime = simple_bench(grid, nsteps)
        except RuntimeError:
            break
        throughput = grid.ncells * nsteps / walltime
        print("{:>10} {:>10.5f} {:>12.3E}".format(grid.ncells, walltime, throughput))
        results.append((grid.ncells, walltime, throughput))
        p += 1

    print()
    return gpu_name, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mumax+ GPU throughput benchmark")
    parser.add_argument("--all", action="store_true",
                        help="benchmark every GPU on the machine, one after another")
    parser.add_argument("--out", help=argparse.SUPPRESS)  # internal: worker writes JSON here
    args = parser.parse_args()

    if args.out:
        # Worker mode: benchmark the single visible GPU and hand results to the parent
        gpu_name, results = run_benchmark(args.nsteps)
        with open(args.out, "w") as f:
            json.dump({"gpu": gpu_name, "results": results}, f)
        sys.exit(0)

    if args.all:
        # Parent mode: one subprocess per GPU
        all_runs = []
        for idx, name in list_gpus():
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=idx)
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "result.json")
                subprocess.run(
                    [sys.executable, os.path.abspath(__file__),
                     "--out", out],
                    env=env,
                )
                if not os.path.exists(out):
                    print(f"WARNING: benchmark failed on GPU {idx} ({name}), skipping")
                    continue
                with open(out) as f:
                    all_runs.append(json.load(f))
    else:
        # Single GPU: the user's choice via CUDA_VISIBLE_DEVICES, or the default GPU
        gpu_name, results = run_benchmark()
        all_runs = [{"gpu": gpu_name, "results": results}]

    if not any(run["results"] for run in all_runs):
        sys.exit("No benchmark results were collected.")

    with open("bench.txt", "w") as file:
        for run in all_runs:
            for ncells, walltime, throughput in run["results"]:
                file.write(f"{ncells}    {walltime}    {throughput}    {run['gpu']}\n")

    for i, run in enumerate(all_runs):
        ncells = [r[0] for r in run["results"]]
        throughputs = [r[2] for r in run["results"]]
        plt.loglog(ncells, throughputs, "-o", label=f"{run['gpu']} (#{i})")
    plt.xlabel("Number of cells")
    plt.ylabel("Throughput (cells/s)")
    plt.legend()
    plt.show()
