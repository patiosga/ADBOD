import time
from turtle import speed
from sklearn.neighbors import KDTree
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from prts import ts_recall, ts_precision
# from TSB_UAD_code.feature import Window

from Techniques.dynamic import dynamic_kr
from New_Dyn import optimized_dynamic
from New_Dyn import parallel_opt, multi_threading
from New_Dyn import annoy_approx_knn
from New_Dyn import hnsw_approx_knn
from New_Dyn import kd_tree
from New_Dyn import faiss_exact
from New_Dyn import all_numba
from New_Dyn import basic_opts
import tsfel
import variables

import pyarrow.csv as pc
import pyarrow as pa


# A Class to run the dynamic method with different parameters and optimization and return the results
class Experiment:

    def __init__(self, slide=100, window=200, filepath = './data/YAHOO/Yahoo_A1real_1_data.out'):

        # Initialize the timeseries to experiment on
        # df: np.ndarray = pd.read_csv(filepath, header=None).dropna().to_numpy()
        # self.data: np.ndarray = df[:, 0].reshape(-1, 1).astype(np.float32)  # Το reshape γίνεται γιατί η cdist δέχεται 2D arrays
        # self.label: np.ndarray = df[:, 1]
        # print('Initial data loaded')

        # Read CSV into a PyArrow Table (zero-copy efficient)
        table = pc.read_csv(filepath)
        arrays = [col.to_numpy() for col in table.columns]
        self.data = arrays[0].reshape(-1, 1).astype(np.float32)
        self.label: np.ndarray = arrays[1]




    def run_dyn_with_z_selected(self, z:list, slide=100, window=200):
        '''
        Run the dynamic method with chosen set of z values
        :param z: The set of z values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = dynamic_kr(slide=slide, window=window)
        # Change the z possible values manually
        dyn.z = z

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    
    def run_dyn_with_ks_selected(self, ks:list, slide=100, window=200):
        '''
        Run the dynamic method with chosen set of k values
        :param ks: The set of k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''
        
        dyn = dynamic_kr(slide=slide, window=window)
        # Change the k possible values manually of the C set
        dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    


# TODO: ΜΠΟΡΟΥΝ ΟΙ ΔΥΟ ΠΑΡΑΠΑΝΩ ΣΥΝΑΡΤΗΣΕΙΣ ΝΑ ΣΥΓΧΩΝΕΥΤΟΥΝ ΣΕ ΑΥΤΗ 
    def run_dyn_with_k_and_z_selected(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the dynamic method with chosen set of k values AND z values
        :param z: The set of z values to run the dynamic method with
        :param ks: The set of k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''
        
        dyn = dynamic_kr(slide=slide, window=window)
        # Change the k possible values manually of the C set
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_optimized(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = optimized_dynamic.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_parallel_opts(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with parallelization on the ks for loop. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = parallel_opt.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_basic_opts(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with the basic code opts (depricated). Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = basic_opts.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_threads_opts(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with multi-threading on the ks for loop. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = multi_threading.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_annoy(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with approximate knn using the Annoy library. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = annoy_approx_knn.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_hnsw(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with approximate knn using the Hnsw library. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = hnsw_approx_knn.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_kdtree(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with exact knn using a kdtree (scipy). Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = kd_tree.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_faiss_exact(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with exact knn using the flat faiss index. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with
        :param ks: The k values to run the dynamic method with
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''

        dyn = faiss_exact.dynamic_kr(slide=slide, window=window)
        if z is not None:
            dyn.z = z
        if ks is not None:
            dyn.k = ks

        # Run the dynamic method
        start = time.time()
        scores: np.ndarray = dyn.fit(self.data)
        end = time.time()
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def run_dyn_clean_numba(self, z:list=None, ks:list=None, slide=100, window=200):
        '''
        Run the optimized dynamic method with paraall function compiled with JIT. Possibillity to change the z and k values
        :param z: The z values to run the dynamic method with (MUST BE A NUMPY ARRAY)
        :param ks: The k values to run the dynamic method with (MUST BE A NUMPY ARRAY)
        :param slide: The slide parameter of the dynamic method
        :param window: The window parameter of the dynamic method
        :return: The scores and the total time the method took
        '''


        if z is not None and ks is not None:
            start = time.time()
            scores = all_numba.fit(self.data, z, ks, slide, window)
            end = time.time()
        elif ks is not None:
            start = time.time()
            scores = all_numba.fit(self.data, z, ks, slide, window)
            end = time.time()
        elif z is not None:
            start = time.time()
            scores = all_numba.fit(self.data, z, ks, slide, window)
            end = time.time()
        else:
            start = time.time()
            scores = all_numba.fit(self.data, z, ks, slide, window)
            end = time.time()
        
        total_time = end - start
        # print(f"Dynamic method took {total_time} seconds")

        return scores, total_time
    

    def post_processing_analytics(self, scores):
        # Calculate the recall, precision and F1 score
        if scores.dtype == np.bool_:
            scores = np.where(scores == True, 1, 0)
        recall = recall_score(self.label, scores, zero_division=1)
        precision = precision_score(self.label, scores, zero_division=1)
        
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)

        return recall, precision, f1
    
    @staticmethod
    def test_dataset(dataset_root_name="./data/YAHOO/Yahoo_A1real_", mode:str = 'z', slide=100, window=200, z:list = None, k:list = None):
        '''
        Test the chosen dataset with the selected mode e.g. for mode='z' test the dataset with the selected z values
        :param dataset_root_name: The root name of the dataset
        :param mode: The mode to test the dataset with --> 1. 'z' for testing with z values, 2. 'k' for testing with k values, 3. 'both' for testing with both z and k values
        4. 'approx' for approximate knn, 5. 'opt' for code optimizations, 6. 'parallel' for parallelized optimizations

        '''

        times = []
        recalls = []
        precisions = []
        f1s = []

        for i in tqdm(range(100)):
            try:
                filepath = f'{dataset_root_name}{i}_data.out'

                # Initialize the experiment object for the specific file in the dataset
                exp = Experiment(filepath=filepath)

            except Exception as e:  # Για να μη χτυπαει για οσα αρχεια δεν υπαρχουν μεχρι το 100
                continue

            if mode == 'z':
                scores, ttime = exp.run_dyn_with_z_selected(z, slide, window)
            elif mode == 'k':
                scores, ttime = exp.run_dyn_with_ks_selected(k, slide, window)
            elif mode == 'both':
                scores, ttime = exp.run_dyn_with_k_and_z_selected(z=z, ks=k, slide=slide, window=window)
            elif mode == 'opt':
                scores, ttime = exp.run_dyn_optimized(z=z, ks=k, slide=slide, window=window)
            elif mode == 'parallel':
                scores, ttime = exp.run_dyn_parallel_opts(z=z, ks=k, slide=slide, window=window)
            elif mode == 'threads':
                scores, ttime = exp.run_dyn_threads_opts(z=z, ks=k, slide=slide, window=window)
            elif mode == 'annoy':
                scores, ttime = exp.run_dyn_annoy(z=z, ks=k, slide=slide, window=window)
            elif mode == 'hnsw':
                scores, ttime = exp.run_dyn_hnsw(z=z, ks=k, slide=slide, window=window)
            elif mode == 'kdtree':
                scores, ttime = exp.run_dyn_kdtree(z=z, ks=k, slide=slide, window=window)
            elif mode == 'faiss':
                scores, ttime = exp.run_dyn_faiss_exact(z=z, ks=k, slide=slide, window=window)
            elif mode == 'numba':
                scores, ttime = exp.run_dyn_clean_numba(z=z, ks=k, slide=slide, window=window)
            elif mode == 'basic_opts':
                scores, ttime = exp.run_dyn_basic_opts(z=z, ks=k, slide=slide, window=window)
            else:
                raise ValueError("Invalid mode")
            
            recall, precision, f1 = exp.post_processing_analytics(scores)

            times.append(ttime)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        return times, recalls, precisions, f1s
    

    @staticmethod
    def test_all_datasets(mode:str = 'z', slide=100, window=200, z:list = None, k:list = None):
        
        df = pd.read_csv('original_results.csv')
        speedups = []
        f1s = []

        # Run code once for jit to compile the kr function
        # for dataset in variables.datasets:
        #     times_new, _, _, f1s_new = Experiment.test_dataset(dataset_root_name=dataset, mode=mode, slide=slide, window=window, z=z, k=k)
        #     break

        # Test the Yahoo dataset with the selected z values
        for dataset in variables.datasets:
            times_new, _, _, f1s_new = Experiment.test_dataset(dataset_root_name=dataset, mode=mode, slide=slide, window=window, z=z, k=k)

            # Original recored times
            original_total_dataset_time = df.groupby('dataset').get_group(dataset)['time'].sum()

            # Calculate speedup
            speedup = original_total_dataset_time / np.sum(times_new)
            speedups.append(speedup)

            # Mean f1 score
            f1s.append(np.mean(f1s_new))

        return speedups, f1s



    @staticmethod
    def plot_results(speedups, f1s, title=None):

        # Get the original mean f1 scores in the same order as the datasets (ίδια σειρά με τα f1s που έρχονται ως όρισμα)
        df = pd.read_csv('original_results.csv')
        original_f1s = []
        for dataset in variables.datasets:
            original_f1s.append(df.groupby('dataset').get_group(dataset)['f1'].mean())

        # Calculate the difference in mean f1 scores
        print(f1s, original_f1s)
        f1_diff = [f1 - original_f1 for f1, original_f1 in zip(f1s, original_f1s)]
        print(f1_diff)

        plt.figure(figsize=(8, 6))
        plt.xlabel('Speedup')
        plt.ylabel('Difference in F1 Score')

        plt.xlim(min(speedups) * 0.9, max(speedups) * 1.1)

        markers = ['o', 's', 'D', '^', 'v', '*', 'p', 'X', '<', '>']
        colors = plt.cm.rainbow(np.linspace(0, 1, len(variables.dataset_names)))

        # Plot scatter points
        for i, dataset in enumerate(variables.dataset_names):
            plt.scatter(speedups[i], f1_diff[i], color=colors[i], label=dataset, marker=markers[i % len(markers)])

        # Fetch updated y-limits after scatter plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Add vertical and horizontal dashed lines
        for i, dataset in enumerate(variables.dataset_names):
            plt.axhline(y=f1_diff[i], color=colors[i], linestyle='--', linewidth=0.5, 
                        xmax=(speedups[i] - x_min) / (x_max - x_min))
            plt.axvline(x=speedups[i], color=colors[i], linestyle='--', linewidth=0.5, 
                        ymax=(f1_diff[i] - y_min) / (y_max - y_min))

        plt.legend()
        plt.title(title)
        plt.show()




if __name__ == "__main__":
    # Run the experiment

    z = np.arange(4, 20) / 2
    k = np.array([5,6,7,8,9,10,13,17,21,30,40])
    
    z1 = np.arange(4,8)
    k1 = np.array([6,7,8])
    

    speedups, f1s = Experiment.test_all_datasets(mode='both', slide=100, window=200, z=z, k=k)
    Experiment.plot_results(speedups, f1s)


