import profile
from typing import final
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import faiss
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor



class dynamic_kr:

    def __init__(self, window_norm=False, slide=100, window=200, policy="or"):
        
        self.z = list(d / 2 for d in range(4, 20))  # ORIGINAL
        self.metric= "euclidean"
        self.k=[5,6,7,8,9,10,13,17,21,30,40]
     
        self.window_norm=window_norm
        self.slide=slide
        self.window=window
        self.policy=policy


    def _calc_dist(self,query : np.ndarray, pts: np.ndarray):
        return cdist(query, pts, metric='cityblock')
        # return np.abs(query[:, None] - query)  # χρησιμοποιειται ευκλειδια αποσταση αλλα έχω μονοδιάστατα σημεία οπότε δεν χρειάζεται να υπολογιστεί η ρίζα του τετραγωνου της διαφορας. Η απολυτη

    def search(self,query: np.ndarray,points:np.ndarray, k: int):
        '''
        Ο πίνακας D είναι  W x W είναι οι αποστασεις του i σημείου/σειράς '''
        dists = self._calc_dist(query,points)

        # I = (
        #     np.argsort(dists, axis=1)
        #     if k > 1
        #     else np.expand_dims(np.argmin(dists, axis=1), axis=1)
        # )
        I = np.argsort(dists, axis=1)  # k is never 1 or less
        D = np.take_along_axis(np.array(dists), I, axis=1)
        return D, I
    

    def search_faiss(self, query: np.ndarray, points:np.ndarray, k: int):
        # Create an index
        index = faiss.IndexFlatL2(1)
        # Add the data to the index
        index.add(points)
        # Find the k nearest neighbors of all the points in the data
        D, I = index.search(points, k)
        return D, I


    def _dynamic_rk(self, query: np.ndarray, pts: np.ndarray):
        ks = [k for k in self.k if k < pts.shape[0] - 1]

        if len(ks) == 0:
            return -1, -1.0, -1.0

        # Save D in memory so as to not calculate it again later
        self.D, _ = self.search(pts, query, max(ks) + 1)

        max_res, k_sel, r_sel = -1.0, -1, -1.0

        def compute_res(k):
            kdists = self.D[:, k]  # k-th (!) closest distance to other points in window
            m, s = kdists.mean(), kdists.std()
            rs = [m + z_i * s for z_i in self.z]

            best_res, best_k, best_r = -1.0, -1, -1.0

            for r in rs:
                if r < 0:
                    continue

                inliers_dists = kdists[kdists <= r]
                n_outliers = (kdists > r).sum()
                if n_outliers == 0 or len(inliers_dists) <= 1:
                    continue

                dmean, dstd = m - inliers_dists.mean(), s - inliers_dists.std()
                res = (dmean / m + dstd / s) / (n_outliers)

                if res > best_res:
                    minoutlier = kdists[kdists > r].min() 
                    maxinlier = kdists[kdists <= r].max()
                    best_res, best_k, best_r = res, k, (minoutlier + maxinlier) / 2

            return best_k, best_r, best_res


        with ThreadPoolExecutor() as executor:
            results = list(executor.map(compute_res, ks))

        for k, r, res in results:
            if res > max_res:
                max_res, k_sel, r_sel = res, k, r
                   

        return k_sel, r_sel, max_res


    def collect_scores(self, query_df):
        '''
        Επιστρέφει το score για κάθε σημείο του query_df -->
        1 αν είναι outlier, 0 αν δεν είναι'''
        pts = query_df

        if self.window_norm:
            pts=(pts-pts.min())/(pts.max()-pts.min())

        curr_k, r, res = self._dynamic_rk(pts, pts)
        #print(f"chosen k:{curr_k} r:{r} with res:{res}")
        # k = curr_k + 1  # ???? αφου το k_sel είναι το k-1

        # D, _ = self.search(pts, pts, k)  # Το κάνει δευτερη φορά ??? αφου γίνεται ήδη μια στο _dynamic_rk????
        

        score = []

        for d in self.D[:, curr_k]:
            if d < 0:
                score.append(0)
            elif d > r and (d - r) / d > 0.05:  # το d > r δεν ειναι περιττό δεδομένου ότι ισχύει το δεύτερο???
                score.append(1)
            else:
                score.append(0)
        
        return score
    

    def combinescores(self, final_score: dict, currentdf: pd.DataFrame, ids: list):
        '''
        Ανάλογα το policy επιλέγει το αν καποιο σημείο είναι ανωμαλο ή όχι
        με βάση τα παράθυρα που έχει βγει ανώμαλο (??)'''

        scores = self.collect_scores(currentdf)

        for sc, ind in zip(scores, ids): 
            # renew the scores of the points in the window and add the new ones in the dictionary final_score
            if ind in final_score:  # update the score of the index depending on the policy
                # if self.policy=="or":
                final_score[ind] |= sc  # bitwise OR
                # elif self.policy=="and":
                #     final_score[ind] = min(final_score[ind], sc)
                # elif self.policy=="first":
                #     final_score[ind] = final_score[ind]
                # elif self.policy=="last":
                #     final_score[ind] = sc
                # else:
                #     final_score[ind] = sc
            else:  # add new index to the dictionary
                final_score[ind] = sc
        return final_score
    

    def fit(self, df):
        final_score = {}  # dictionary with key: index of the point, value: score of the point 0 or 1 (inlier or outlier)
        pos = 0
        for pos in range(self.window, len(df), self.slide):
            # print(f"pos:{pos}, window len:{self.window}, df len:{len(df)}")
            # pos --> is the position of the last element of the current window
            # Currentdf is the current window that slides along the data given as df
            currentdf = df[max(pos - self.window, 0) : pos]  # ??? pos - window will never be negative ???
            ids = [idx for idx in range(max(pos-self.window, 0), pos)]

            final_score = self.combinescores(final_score, currentdf, ids)

        if pos < len(df): # If there are still some elements left repeat process for the leftovers
            pos = len(df)
            currentdf = df[max(pos - self.window, 0) : pos]
            ids = [idx for idx in range(max(pos - self.window, 0), pos)]
            final_score = self.combinescores(final_score, currentdf, ids)  
            # για 'or' policy μπορει πιο ευκολα να γινει ενα συνολικό bitwise OR μεταξύ όλων των scores 
            # των παραθύρων που έχουν περαστεί  (?!!!!!!!!!!!!!!!!!!!!!????)

        scores_to_return=list(final_score[idx] for idx in range(0, len(df)))

        return np.array(scores_to_return)  # 1s and 0s for each point in the series
    


if __name__ == '__main__':
    pass 