import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from TSB_UAD.models.feature import Window
from TSB_UAD.models.sand import SAND

from TSB_UAD.utils.slidingWindows import find_length, plotFig

datasetname=f'../../data/YAHOO/Yahoo_A1real_{3}_data.out'

df = pd.read_csv(datasetname, header=None).dropna().to_numpy()
max_length=len(df)+1

data = df[:max_length,0].astype(float)
label = df[:max_length,1]

slidingWindow = find_length(data)

#print(f" sequence size: {slidingWindow}")
#self.slidingWindow = 1
X_data = Window(window = slidingWindow).convert(data).to_numpy()
actual_slideing=12
modelName='SAND (offline)'
clf = SAND(pattern_length=actual_slideing,subsequence_length=4*(actual_slideing))
x = X_data
clf.fit(x,online=True,alpha=0.5,init_length=500,batch_size=200,verbose=True,overlaping_rate=int(4*actual_slideing))
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
plotFig(data, label, score, slidingWindow, fileName="testtest", modelName=modelName)
plt.show()