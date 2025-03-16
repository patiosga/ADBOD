import numpy as np
import scipy.spatial.distance as ssd
import timeit
import matplotlib.pyplot as plt
from numba import njit, prange
import multiprocessing
from joblib import Parallel, delayed

# Get number of available CPU cores
num_cores = multiprocessing.cpu_count()

# Numba-optimized cdist implementation with parallelization
@njit(parallel=True)
def numba_cdist(XA, XB):
    m, n = XA.shape[0], XB.shape[0]
    dists = np.empty((m, n), dtype=np.float64)
    
    for i in prange(m):  # Parallel loop
        for j in range(n):
            dists[i, j] = np.sqrt(np.sum((XA[i] - XB[j]) ** 2))
    
    return dists


def parallel_cdist(XA, XB, metric="euclidean"):
    n_jobs = -1  # Use all available CPU cores
    m = XA.shape[0]
    
    def compute_row(i):
        return ssd.cdist(XA[i:i+1], XB, metric=metric)[0]

    return np.array(Parallel(n_jobs=n_jobs)(delayed(compute_row)(i) for i in range(m)))


# Function to benchmark execution time
def benchmark(func, XA, XB, num_runs=3):
    times = timeit.repeat(lambda: func(XA, XB), number=1, repeat=num_runs)
    return np.mean(times), np.std(times)

# Generate random datasets for testing
sizes = [100, 500, 1000, 2000]  # Number of points
dim = 3  # 3D points
scipy_times_single, scipy_times_parallel, numba_times = [], [], []

for size in sizes:
    XA = np.random.rand(size, dim)
    XB = np.random.rand(size, dim)
    
    # Benchmark SciPy's single-threaded cdist
    scipy_time_single, _ = benchmark(lambda XA, XB: parallel_cdist(XA, XB), XA, XB)
    scipy_times_single.append(scipy_time_single)

    # Benchmark SciPy's multi-threaded cdist
    scipy_time_parallel, _ = benchmark(lambda XA, XB: ssd.cdist(XA, XB, metric='euclidean', workers=num_cores), XA, XB)
    scipy_times_parallel.append(scipy_time_parallel)

    # Benchmark Numba's parallel cdist
    numba_time, _ = benchmark(numba_cdist, XA, XB)
    numba_times.append(numba_time)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(sizes, scipy_times_single, label="SciPy cdist (Single-core)", marker="o", linestyle="--")
plt.plot(sizes, scipy_times_parallel, label=f"SciPy cdist ({num_cores} Cores)", marker="o")
plt.plot(sizes, numba_times, label="Numba Parallel cdist", marker="s")
plt.xlabel("Number of Points")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.title("Execution Time Comparison: SciPy Single vs Multi-core vs Numba Parallel")
plt.grid()
plt.show()

