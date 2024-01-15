import matplotlib.pyplot as plt
import pandas as pd
from Techniques.dynamic import dynamic_kr


def checkdynkrOnStatic():
    df = pd.read_csv("data/dataOoutliers.txt", header=None)
    df = df.iloc[:len(df.index) // 5]
    df = df.dropna()
    plt.scatter(df.values[:, 0], df.values[:, 1])
    plt.show()

    normalized_df = (df - df.min()) / (df.max() - df.min())
    data = normalized_df.values

    print(df.head())
    clf = dynamic_kr(slide=10, window=len(normalized_df.index) + 100, window_norm=False, policy="or")

    del df

    score = clf.fit(data)
    labels = [sc > 0 for sc in score]
    plt.plot(score)
    plt.show()
    plt.scatter(data[:, 0], data[:, 1])
    reds = data[labels]
    plt.scatter(reds[:, 0], reds[:, 1], color="red")
    plt.show()


if __name__ == "__main__" :
    checkdynkrOnStatic()



