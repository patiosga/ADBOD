import dis
import numpy as np
from scipy.spatial.distance import cdist

test = np.random.randn(200, 1)

dists = cdist(test, test, 'euclidean')
print(dists.shape)

ks = np.array([1, 2, 3])

import time
start = time.time()
for i in range(10000):
    dists_temp = np.sort(dists, axis=1)


    dists_temp2 = np.empty((dists.shape[0], dists.shape[0]), dtype=float)
    dists_temp2[:, ks] = np.partition(dists, ks, axis=1)[:, ks]

    if not np.allclose(dists_temp[:, ks], dists_temp2[:, ks], atol=0):
        print('Error')
end = time.time()
print(end - start)

# check if the two methods are the same
print(np.allclose(dists_temp[:, ks], dists_temp2[:, ks], atol=0))



