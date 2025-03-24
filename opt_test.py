from line_profiler import LineProfiler
from New_Dyn.optimized_dynamic import dynamic_kr
import numpy as np
import pandas as pd


filepath = './data/YAHOO/Yahoo_A1real_2_data.out'
df: np.ndarray = pd.read_csv(filepath, header=None).dropna().to_numpy()
data: np.ndarray = df[:, 0].astype(float).reshape(-1, 1)

# Create an instance of the dynamic_kr class
clf = dynamic_kr(slide=100, window=200, window_norm=False, policy="or")

lp = LineProfiler()
lp.add_function(clf.collect_scores)
lp.add_function(clf.combinescores)
lp.add_function(clf.search)
lp.add_function(clf._calc_dist)
lp.add_function(clf._dynamic_rk)
lp.add_function(clf.fit)
lp.enable()
clf.fit(data)
lp.disable()
lp.print_stats()

# 98% του χρονου στο combine scores!!!!!!!
