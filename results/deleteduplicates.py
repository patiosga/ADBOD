import pandas as pd


def deleteduplicate(name="results.csv"):
    dfdyn = pd.read_csv(name,header=0)

    normlist=[True if kati else False for kati in dfdyn["normalize"]]
    dfdyn["normalize"]=normlist
    dfdyn["initSubseq"] = [None if kati == 'None' else kati for kati in dfdyn["initSubseq"]]
    dfdyn["initSubseq"]=[float(kati) if kati is not None else None for kati in dfdyn["initSubseq"]]

    dfdyn["features"]=['False' if kati=='0' else kati for kati in dfdyn["features"]]
    dfdyn["features"]=['False' if kati==False else kati for kati in dfdyn["features"]]
    dfdyn["features"]=['True' if kati==True else kati for kati in dfdyn["features"]]
    dfdyn["features"]=['True' if kati=='1' else kati for kati in dfdyn["features"]]
    print(dfdyn["features"].unique())
    print(len(dfdyn.index))
    dfdyn=dfdyn.drop_duplicates(subset=["method","k","r","window","shift","initSubseq","features","dataset","normalize"], keep='last')
    dfdyn.to_csv(name,index=False)
    print(len(dfdyn.index))

deleteduplicate(name="results_final.csv")
deleteduplicate(name="resultsKR_final.csv")
#dfdyn = pd.read_csv("resultsKR.csv",header=0)
#dfdyn = pd.read_csv("results.csv",header=0)

#dfdyn=dfdyn[dfdyn["normalize"]==True]

#print(dfdyn["features"].unique())
