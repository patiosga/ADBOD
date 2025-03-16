import profile
import stat
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import faiss
from numba import njit, prange


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
        return cdist(query, pts, metric=self.metric)
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
        I = np.argpartition(dists, k, axis=1)[:, :k]  # Indices of k smallest distances --> !!! Quickselect algorithm --> O(n) expected
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
    
    
    @njit(parallel=True, fastmath=True)
    def _dynamic_rk(self, query: np.ndarray, pts: np.ndarray):
        ks = [k for k in self.k if k < pts.shape[0] - 1]

        if len(ks) == 0:
            return -1, -1.0, -1.0

        # Save D in memory so as to not calculate it again later
        self.D, _ = self.search(pts, query, max(ks) + 1)

        max_res, k_sel, r_sel = -1.0, -1, -1.0

        results = np.zeros((len(ks), 3))  # Store results to find the best later

        for idx in prange(len(ks)):
            k = ks[idx]
            kdists:np.ndarray = self.D[:, k]  # k-th (!) closest distance to other points in window is at k position (and not k-1) because 0 position is the point itself

            kdistmin=kdists.min()
            kdistmax=kdists.max()
            # Normalize the distances
            kdists=(kdists-kdistmin)/(kdistmax-kdistmin)


            m = kdists.mean()
            s = kdists.std()
            # Test different z values
            for z_i in self.z:
                # Calculate the radius
                r = m + z_i * s

                inliers_dists = kdists[kdists <= r]  # distances that are less than r for each point
                n_outliers = (kdists > r).sum()  # total number of outliers for a k, r pair in the window
                if r < 0:
                    continue
                if n_outliers == 0:
                    break  # because the objective function divides by n_outliers

                # Βρίσκει απόσταση του min outlier και max inliers ώστε τελικα να πάρει ακτίνα R που είναι στη μέση τους
                # Δοκιμή χωρίς αυτούς τους υπολογισμούς και επιλογη r = m + z_i * s με το z_i που δίνει το μεγαλύτερο res --> απέτυχε παταγωδώς
                minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin
                r = (minoutlier + maxinlier) / 2  #--> τελικά θα το κάνω έτσι αλλα αυτοί οι υπολογισμοί θα γίνονται μόνο αν res > max_res

                if len(inliers_dists) <= 1:
                    continue

                dmean = m - inliers_dists.mean()
                dstd = s - inliers_dists.std()
                res = (dmean / m + dstd / s) / ((n_outliers))

                if res > max_res:
                    # check if objective function gives higher output than 
                    # the previous best and keep the best selected k, r, and result

                    # This was being calculated in the loop but it can be done only if res > max_res
                    minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                    maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin

                    k_sel, r_sel, max_res = k, (minoutlier+maxinlier)/2, res
                
            results[idx] = [k_sel, r_sel, max_res]

        
        # Find the best k, r, and max_res
        best_idx = np.argmax(results[:, 2])
        k_sel, r_sel, max_res = results[best_idx]

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

        # ΑΠΟΚΛΕΙΕΤΑΙ ΑΥΤΟ ΝΑ ΜΗ ΓΙΝΕΤΑΙ ΠΙΟ ΓΡΗΓΟΡΑ !!!!!
        for d in self.D[:, curr_k]:
            if d < 0 or r < 0:
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
            if ind in final_score.keys():  # update the score of the index depending on the policy
                if self.policy=="or":
                    final_score[ind] = max(final_score[ind], sc)
                elif self.policy=="and":
                    final_score[ind] = min(final_score[ind], sc)
                elif self.policy=="first":
                    final_score[ind] = final_score[ind]
                elif self.policy=="last":
                    final_score[ind] = sc
                else:
                    final_score[ind] = sc
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

        scores_to_return=[]
        for idx in range(0, len(df)):
            scores_to_return.append(final_score[idx])

        return np.array(scores_to_return)  # 1s and 0s for each point in the series
    


if __name__ == "__main__":

    pass