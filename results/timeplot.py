import math
import statistics
import matplotlib.pyplot as plt
import pandas as pd


df1=pd.read_csv("times.txt",header=0)
print(df1.head())

df=df1[df1["slide"]==200]
print(df.head(10))
print(df["sublength"].values)

plt.subplot(121)
plt.plot(df["sublength"].values,df["dyntime"].values,marker="D",linewidth=3,markersize=15)
plt.plot(df["sublength"].values,df["statictime"].values,"-o",linewidth=3,markersize=15)
plt.ylabel("time (s)")
plt.xlabel("sub-sequence size")



plt.subplot(122)

df=df1[df1["sublength"]==50]
print(df.head(10))
print(df["window"].values)
plt.plot(df["window"].values,df["dyntime"].values,marker="D",linewidth=3,markersize=15,label="Dyn")
plt.plot(df["window"].values,df["statictime"].values,"-o",linewidth=3,markersize=15,label="DOD")
plt.ylabel("time (s)")
plt.xlabel("window size")

plt.show()