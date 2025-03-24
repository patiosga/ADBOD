import profile
from typing import final
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from bitarray import bitarray



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
        D = np.sort(dists, axis=1)
        # I = np.argsort(dists, axis=1)
        return D, None



    def _dynamic_rk(self, query: np.ndarray, pts: np.ndarray):
        ks = [k for k in self.k if k < pts.shape[0] - 1]
        
        if len(ks) == 0:
            return -1, -1.0, -1.0

        # Save D in memory so as to not calculate it again later
        self.D, _ = self.search(pts, query, ks[-1] + 1)  # εδώ είχε max(ks) + 1 αλλά αυτό είναι το τελευταίο k που θα χρειαστεί!!!!!!

        max_res, k_sel, r_sel = -1.0, -1, -1.
        
        # Προϋπολογισμός μέσης τιμής και τυπικής απόκλισης για κάθε k για χρήση vectorized υπολογισμών
        means = self.D[:, ks].mean(axis=0)
        stds = self.D[:, ks].std(axis=0)

        for i, k in enumerate(ks):           
            kdists: np.ndarray = self.D[:, k]  # k-th (!) closest distance to other points in window

            # kdistmin=kdists.min()
            # kdistmax=kdists.max()
            # # Normalize the distances
            # kdists=(kdists-kdistmin)/(kdistmax-kdistmin)

            # m = kdists.mean()
            # s = kdists.std()
            m = means[i]
            s = stds[i]
           
            # Test different z values
            rs = [m + z_i * s for z_i in self.z]

            for r in rs:
                inliers_dists = kdists[kdists <= r]  # distances that are less than r for each point
                n_outliers = (kdists > r).sum()  # total number of outliers for a k, r pair in the window
                if r < 0:
                    continue
                if n_outliers == 0:
                    break  # because the objective function divides by n_outliers

                # Βρίσκει απόσταση του min outlier και max inliers ώστε τελικα να πάρει ακτίνα R που είναι στη μέση τους
                # Δοκιμή χωρίς αυτούς τους υπολογισμούς και επιλογη r = m + z_i * s με το z_i που δίνει το μεγαλύτερο res --> απέτυχε παταγωδώς
                # minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                # maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin
                # r = (minoutlier + maxinlier) / 2 --> τελικά θα το κάνω έτσι αλλα αυτοί οι υπολογισμοί θα γίνονται μόνο αν res > max_res

                if len(inliers_dists) <= 1:
                    continue

                dmean = m - inliers_dists.mean()
                dstd = s - inliers_dists.std()
                
                res = (dmean / m + dstd / s) / (n_outliers)
                
                if res > max_res:
                    # check if objective function gives higher output than 
                    # the previous best and keep the best selected k, r, and result

                    # This was being calculated in the loop but it can be done only if res > max_res
                    # minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                    # maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin

                    # Without normalization και χωρίς επανυπολογισμό των inliers (το να μην ξαναυπολογίσω outliers είναι πιο αργό για κάποιον λόγο που μάλλον έχει να κάνει με το ότι το inliers ξαναχρηισμοποιείται πιο πάνω)
                    minoutlier = min(kdists[kdists > r]) 
                    maxinlier = max(inliers_dists)

                    max_res, k_sel, r_sel = res, k, (minoutlier+maxinlier)/2
                   

        return k_sel, r_sel, max_res


    def collect_scores(self, query_df):
        '''
        Επιστρέφει το score για κάθε σημείο του query_df -->
        1 αν είναι outlier, 0 αν δεν είναι'''
        pts = query_df

        # if self.window_norm:
        #     pts=(pts-pts.min())/(pts.max()-pts.min())

        curr_k, r, _ = self._dynamic_rk(pts, pts)
        #print(f"chosen k:{curr_k} r:{r} with res:{res}")
        # k = curr_k + 1  # ???? αφου το k_sel είναι το k-1

        # D, _ = self.search(pts, pts, k)  # Το κάνει δευτερη φορά ??? αφου γίνεται ήδη μια στο _dynamic_rk????
        

        # score = []

        # for d in self.D[:, curr_k]:
        #     if d < 0:
        #         score.append(0)
        #     elif d > r and (d - r) / d > 0.05:  # το d > r δεν ειναι περιττό δεδομένου ότι ισχύει το δεύτερο???
        #         score.append(1)
        #     else:
        #         score.append(0)

        d_values = self.D[:, curr_k]

        # Αποφυγή διαίρεσης με 0
        safe_mask = d_values != 0
        ratio = np.zeros_like(d_values)  # numpy array με μηδενικά ίδιου μεγέθους με το d_values
        ratio[safe_mask] = (d_values[safe_mask] - r) / d_values[safe_mask]  # Υπολογίζω αποτέλεσμα μόνο για τα στοιχεία που δεν είναι 0

        # Vectorized computation
        score = np.where((d_values > r) & (ratio > 0.05), 1, 0)

        # Handle cases where d < 0 or r < 0
        score[(d_values < 0) | (r < 0)] = 0
        
        return score
    

    def combinescores(self, final_score: dict, currentdf: pd.DataFrame, ids: list):
        '''
        Ανάλογα το policy επιλέγει το αν καποιο σημείο είναι ανωμαλο ή όχι
        με βάση τα παράθυρα που έχει βγει ανώμαλο (??)'''

        scores = self.collect_scores(currentdf)
        # Using only or policy!!!
        np.put(final_score, ids, final_score[ids] | scores)  # bitwise OR

        # for sc, ind in zip(scores, ids): 
        #     final_score[ind] |= sc  # bitwise OR
            # renew the scores of the points in the window and add the new ones in the dictionary final_score
            # if ind in final_score:  # update the score of the index depending on the policy
                # if self.policy=="or":
                #   final_score[ind] |= sc  # bitwise OR
                # elif self.policy=="and":
                #     final_score[ind] = min(final_score[ind], sc)
                # elif self.policy=="first":
                #     final_score[ind] = final_score[ind]
                # elif self.policy=="last":
                #     final_score[ind] = sc
                # else:
                #     final_score[ind] = sc
            # else:  # add new index to the dictionary
            #     final_score[ind] = sc
        return final_score
    

    def fit(self, df):
        final_score = np.zeros(len(df), dtype=int)  # Initialize the final score array as all zeros
        pos = 0
        for pos in range(self.window, len(df), self.slide):
            # print(f"pos:{pos}, window len:{self.window}, df len:{len(df)}")
            # pos --> is the position of the last element of the current window
            # Currentdf is the current window that slides along the data given as df
            currentdf = df[(pos - self.window) : pos]  # ??? pos - window will never be negative ???
            # ids = [idx for idx in range(max(pos-self.window, 0), pos)]
            ids = np.arange(pos - self.window, pos)

            final_score = self.combinescores(final_score, currentdf, ids)

        if pos < len(df): # If there are still some elements left repeat process for the leftovers
            pos = len(df)
            currentdf = df[pos - self.window : pos]
            # ids = [idx for idx in range(max(pos - self.window, 0), pos)]
            ids = np.arange(pos - self.window, pos)

            final_score = self.combinescores(final_score, currentdf, ids)  
            # για 'or' policy μπορει πιο ευκολα να γινει ενα συνολικό bitwise OR μεταξύ όλων των scores 
            # των παραθύρων που έχουν περαστεί  (?!!!!!!!!!!!!!!!!!!!!!????)

        # scores_to_return=list(final_score[idx] for idx in range(0, len(df)))

        return final_score  # 1s and 0s for each point in the series
    


