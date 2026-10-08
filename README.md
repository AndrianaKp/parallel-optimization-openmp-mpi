# Parallel Optimization & Hyperparameter Search (OpenMP, MPI, Python)

Parallelisation of two embarrassingly parallel workloads, with speedup measurements, for the *Software & Programming for High-Performance Systems* course (University of Patras, CEID). Team project (2 people).

![C](https://img.shields.io/badge/C-A8B9CC?logo=c&logoColor=black)
![OpenMP](https://img.shields.io/badge/OpenMP-grey)
![MPI](https://img.shields.io/badge/MPI-grey)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

## 1. Multistart global optimization (C)

Find the global minimum of the 4-D **Rosenbrock** function by running a local optimizer (Torczon's **Multidirectional Search**) from 64 random starting points.

| File | Approach |
|---|---|
| `multistart_mds_seq.c` | Sequential baseline |
| `multistart_mds_omp.c` | OpenMP `parallel for`; `critical` for the global best, `atomic` for the evaluation counter |
| `multistart_mds_tasks.c` | OpenMP **tasks**: one task per trial, dynamic scheduling |
| `multistart_mds_mpi.c` | MPI: cyclic distribution of trials, `MPI_Reduce` with `MPI_MINLOC` to find the winning rank, point sent to rank 0 |
| `torczon.c` | Local optimizer (MDS) |

| Workers | OpenMP for | OpenMP tasks | MPI |
|---|---|---|---|
| 1 | 121.3 s (1.00×) | 120.7 s (1.00×) | 119.5 s (1.00×) |
| 2 | 61.5 s (1.97×) | 60.5 s (2.00×) | 61.3 s (1.95×) |
| 4 | 29.0 s (**4.17×**) | 28.8 s (**4.19×**) | 28.9 s (**4.14×**) |

All three versions scale almost linearly and return the same minimum and the same number of function evaluations, which confirms that race conditions are handled correctly.

```bash
cd multistart-optimization
gcc -O2 -fopenmp multistart_mds_omp.c torczon.c -lm -o multistart_omp
OMP_NUM_THREADS=4 ./multistart_omp

mpicc -O2 multistart_mds_mpi.c torczon.c -lm -o multistart_mpi
mpirun -np 4 ./multistart_mpi
```

## 2. Parallel random search for ML hyperparameters (Python)

Tune the regularisation strength **C** of an L1 logistic regression (`saga` solver) with 32 random-search trials, `C ~ 10^U[-5,-1]`, on a synthetic dataset of 100,000 samples × 200 features.

| Workers | `concurrent.futures` | `mpi4py` |
|---|---|---|
| 1 | 73.0 s | 59.8 s |
| 2 | 32.0 s | 29.2 s |
| 4 | 22.9 s (**2.72×**) | 21.0 s (**2.97×**) |

Sequential baseline: 62.4 s. Every version finds the same best model (C = 8.29e-3, validation accuracy 0.8633). MPI scales slightly better because of lower process-management overhead.

```bash
cd hyperparameter-search
pip install numpy scikit-learn mpi4py
python random_search_seq.py
python random_search_futures.py
mpirun -np 4 python random_search_mpi.py
```

## Environment

Experiments were run on Linux (WSL). The `usleep(100)` in the objective function is intentional: it adds artificial work so that speedup is measurable.
