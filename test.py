import numpy as np
from scipy.spatial.distance import cdist


points = np.array([5,1,9,4])
ks = np.array([1,2])

dists = np.empty((points.shape[0], points.shape[0]), dtype=np.float32)
points = points.flatten()
# Pairwise distances
for i in range(points.shape[0]):
    for j in range(i, points.shape[0]):
        dists[i, j] = np.abs(points[i] - points[j])  # Manhattan distance for multi-dimensional 
        dists[j, i] = dists[i, j]

print(dists)
D = np.empty((dists.shape[0], ks.shape[0]), dtype=np.float32)
for i in range(dists.shape[0]):
    D[i, :] = np.partition(dists[i, :], ks)[ks]  # Ταξινόμηση κάθε σειράς ξεχωριστά γιατί η numba δεν υποστηρίζει args στην np.sort
print(D)

   





