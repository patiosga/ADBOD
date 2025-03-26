import profile
from typing import final
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from bitarray import bitarray
from numba import jit, prange, njit
import bisect



    
@njit()
def fit(df:np.ndarray, z: np.ndarray, ks: np.ndarray, slide:int, window:int):
        final_score = np.zeros(len(df), dtype=np.bool_)  # Initialize the final score array as all zeros
        pos = 0
        for pos in range(window, len(df), slide):
            # print(f"pos:{pos}, window len:{window}, df len:{len(df)}")
            # pos --> is the position of the last element of the current window
            # Currentdf is the current window that slides along the data given as df
            currentdf = df[(pos - window) : pos]  # ??? pos - window will never be negative ???
            # ids = [idx for idx in range(max(pos-window, 0), pos)]
            ids = np.arange(pos - window, pos)

            final_score = combinescores(final_score, currentdf, ids, ks, z)

        if pos < len(df): # If there are still some elements left repeat process for the leftovers
            pos = len(df)
            currentdf = df[pos - window : pos]
            # ids = [idx for idx in range(max(pos - window, 0), pos)]
            ids = np.arange(pos - window, pos)

            final_score = combinescores(final_score, currentdf, ids, ks, z)  
            # για 'or' policy μπορει πιο ευκολα να γινει ενα συνολικό bitwise OR μεταξύ όλων των scores 
            # των παραθύρων που έχουν περαστεί  (?!!!!!!!!!!!!!!!!!!!!!????)

        # scores_to_return=list(final_score[idx] for idx in range(0, len(df)))

        return final_score  # 1s and 0s for each point in the series

@njit
def combinescores(final_score: np.ndarray, currentdf: np.ndarray, ids: np.ndarray, ks, z):
        '''
        Ανάλογα το policy επιλέγει το αν καποιο σημείο είναι ανωμαλο ή όχι
        με βάση τα παράθυρα που έχει βγει ανώμαλο (??)'''

        scores = collect_scores(currentdf, ks, z)
        # Using only or policy!!!
        # np.put(final_score, ids, final_score[ids] | scores)  # bitwise OR
        final_score[ids] |= scores  # bitwise OR

        return final_score

@njit(fastmath=True)
def collect_scores(query_df, ks, z):
        '''
        Επιστρέφει το score για κάθε σημείο του query_df -->
        1 αν είναι outlier, 0 αν δεν είναι'''

        D = search(query_df, ks)

        results_kr = _dynamic_rk(query_df, ks, z, D)
        curr_k = results_kr[0]
        r = results_kr[1]
    
        # Find the index/order of the k in the ks array so we can get the corresponding distances from the D matrix
        k_index = np.searchsorted(ks, int(curr_k), side='left')  # Βρίσκει το index του k στον πίνακα self.k που είναι ταξινομημένος οπότε η αναζήτηση γίνεται με binary search
        score = compute_scores(D[:, k_index], r)
        
        return score

@njit()
def _dynamic_rk(pts, ks, z, D):
        '''
        Επιστρέφει το καλύτερο k και r για το query
        :param pts: το παράθυρο που θέλουμε να εξετάσουμε
        :param ks: τα k που θέλουμε να εξετάσουμε
        :param z: οι παράμετροι 
        :param D: Οι αποστάσεις των σημείων μεταξύ τους (μόνο k-οστές αποστάσεις) --> W x len(ks)
        :return: το καλύτερο k και r
        '''

        ks = ks[ks < pts.shape[0] - 1]  # Τα k που είναι μικρότερα από το μέγεθος του παραθύρου
        
        if len(ks) == 0:
            return np.array((0.0, -1.0, -1.0), dtype=np.float32)

        results_kr = compute_best_kr(ks, D, z)

        return results_kr  # κανω float το k γιατί η numba δεν υποστηρίζει int64 μαζί με float64 στο ίδιο tuple


@njit(fastmath=True)
def compute_best_kr(ks: np.ndarray, D: np.ndarray, z:np.ndarray):
    max_res, k_sel, r_sel = -1.0, 0, -1.0

    # Δεν γίνεται με vectorization εδώ επειδή δεν το υποστηρίζει η numba
    means = np.empty(len(ks), dtype=np.float64)
    stds = np.empty(len(ks), dtype=np.float64)
    for i in range(len(ks)):
        # k = ks[i]
        # kdists = D[:, k]
        kdists = D[:, i]
        means[i] = kdists.mean()
        stds[i] = kdists.std()
    # means, stds = compute_means_stds(ks, D)

    for i in range(len(ks)):
        k = ks[i]
        # kdists = D[:, k]
        kdists = D[:, i]  # D is now a W x k matrix and not a W x W where k is the number of possible k values

        m = means[i]
        s = stds[i]

        for j in range(len(z)):
            r = m + z[j] * s
            inliers_dists = kdists[kdists <= r]
            n_outliers = (kdists > r).sum()

            if r < 0:
                continue
            if n_outliers == 0:
                break

            dmean = m - inliers_dists.mean()
            dstd = s - inliers_dists.std()
            
            res = (dmean / m + dstd / s) / (n_outliers)
            
            if res > max_res:
                minoutlier = min(kdists[kdists > r]) # για καποιον πολύ ενδιαφέρον λόγο αυτό είναι πιο γρήγορο από το min(outliers) όταν το outliers έχει υπολογισττεί προηγουμένως όπως το inliers
                maxinlier = max(inliers_dists)

                max_res, k_sel, r_sel = res, k, (minoutlier+maxinlier)/2


    return np.array((float(k_sel), r_sel, max_res), dtype=np.float32)  # κανω float το k γιατί η numba δεν υποστηρίζει int64 μαζί με float64 στο ίδιο tuple
    


@njit(parallel=True)    
def compute_scores(kdists: np.ndarray, r: float):
    score = np.empty(kdists.shape[0], dtype=np.bool_)
    for i in prange(len(kdists)):
        d = kdists[i]
        if d < 0 or r < 0:
            score[i] = False
        else:
            score[i] = (d > r and (d - r) / d > 0.05)

    return score



@njit(parallel=True)
def search(points: np.ndarray, ks: np.ndarray):
    dists = np.empty((points.shape[0], points.shape[0]), dtype=np.float32)
    points = points.flatten()
    # Pairwise distances
    for i in prange(points.shape[0]):
        for j in range(i, points.shape[0]):
            dists[i, j] = np.abs(points[i] - points[j])  # Manhattan distance for multi-dimensional points
            dists[j, i] = dists[i, j]

    # Ταξινόμηση κάθε σειράς ξεχωριστά γιατί η numba δεν υπστηριζεί args στην np.sort
    for i in prange(dists.shape[0]):
        dists[i, :] = np.partition(dists[i, :], ks)

    D = dists[:, ks]  # κρατάω μόνο τις στήλες για τις οποίες ενδιαφέρομαι, δηλαδή τα k που έχω επιλέξει

    return D


# @njit(parallel=True, fastmath=True)
# def compute_means_stds(ks: np.ndarray, D: np.ndarray):
#      # Δεν γίνεται με vectorization εδώ επειδή δεν το υποστηρίζει η numba
#     means = np.empty(len(ks), dtype=np.float64)
#     stds = np.empty(len(ks), dtype=np.float64)
#     for i in prange(len(ks)):
#         # k = ks[i]
#         # kdists = D[:, k]
#         kdists = D[:, i]
#         means[i] = kdists.mean()
#         stds[i] = kdists.std()

#     return means, stds

