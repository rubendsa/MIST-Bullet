from mpi4py import MPI 
import os, subprocess, sys 
import numpy as np 

env = os.environ.copy()
env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
)
n = 4 #threads
args = ["mpirun", "-np", str(n)]

args += [sys.executable, "mpi_runner.py"]

subprocess.check_call(args, env=env)
sys.exit()
