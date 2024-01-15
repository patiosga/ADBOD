import statistics

import pandas as pd

def get_best_on_all_datasets(dfd,parmsspecific={},targetfield="f1Ex+Rpr"):

    alldatasets = dfd["dataset"].unique()

    #temp = alldatasets[0]
    # method,k,r,window,shift,initSubseq,ActSubSeq,Existence,Rrecall,Rprecision,f1Ex+Rpr,f1R+R,features,dataset
    # combinations = k,r,window,shift,initSubseq,features
    Parametercolumns = ["k", "r", "window", "shift", "initSubseq", "features", "normalize"]
    ParamValues = []
    combinations = []
    for param in Parametercolumns:
        if param in parmsspecific.keys():
            ParamValues.append(parmsspecific[param])
        else:
            ParamValues.append([vl for vl in dfd[param].unique()])

    for k in ParamValues[0]:
        for r in ParamValues[1]:
            for window in ParamValues[2]:
                for shift in ParamValues[3]:
                    for initSubseq in ParamValues[4]:
                        for features in ParamValues[5]:
                            for normalize in ParamValues[6]:
                                combinations.append(
                                    {"k": k, "r": r, "window": window, "shift": shift, "initSubseq": initSubseq,
                                     "features": features, "normalize": normalize})
    print(f"combinations: {len(combinations)}")
    # for each combination calculate median ?
    score_per_combination = []
    returnList = []
    for comb in combinations:
        # print()
        try:
            f1list = []
            for datasettemp in alldatasets:
                row = dfd.loc[(dfd["k"] == comb["k"]) & (dfd["r"] == comb["r"]) & (dfd["window"] == comb["window"])
                              & (dfd["shift"] == comb["shift"]) & (dfd["initSubseq"] == comb["initSubseq"]) & (
                                          dfd["features"] == comb["features"])
                              & (dfd["dataset"] == datasettemp) & (dfd["normalize"] == comb["normalize"])]
                f1list.append(row.iloc[-1][targetfield])
                # print(row)
            returnList.append(f1list)
            score_per_combination.append(statistics.median(f1list))
        except Exception as e:
            # print(row)
            # print(comb)
            # print(e)
            returnList.append(f1list)
            score_per_combination.append(0)

    return combinations[score_per_combination.index(max(score_per_combination))],max(score_per_combination),returnList[score_per_combination.index(max(score_per_combination))]

def get_best_on_dataset(dfd,dataset,parmsspecific={},targetfield="f1Ex+Rpr"):
    dfd=dfd[dfd["dataset"].str.contains(dataset)]

    alldatasets=dfd["dataset"].unique()

    temp=alldatasets[0]
    # method,k,r,window,shift,initSubseq,ActSubSeq,Existence,Rrecall,Rprecision,f1Ex+Rpr,f1R+R,features,dataset
    # combinations = k,r,window,shift,initSubseq,features
    Parametercolumns=["k","r","window","shift","initSubseq","features","normalize"]
    ParamValues=[]
    combinations=[]
    for param in Parametercolumns:
        if param in parmsspecific.keys():
            ParamValues.append(parmsspecific[param])
        else:
            ParamValues.append([vl for vl in dfd[param].unique()])

    for k in ParamValues[0]:
        for r in ParamValues[1]:
            for window in ParamValues[2]:
                for shift in ParamValues[3]:
                    for initSubseq in ParamValues[4]:
                        for features in ParamValues[5]:
                            for normalize in ParamValues[6]:
                                combinations.append({"k":k,"r":r,"window":window,"shift":shift,"initSubseq":initSubseq,"features":features,"normalize":normalize})
    print(f"combinations: {len(combinations)}")
    # for each combination calculate median ?
    score_per_combination=[]
    returnList = []
    for comb in combinations:
        #print()
        try:
            f1list = []
            for datasettemp in alldatasets:
                row= dfd.loc[(dfd["k"] == comb["k"]) & (dfd["r"] == comb["r"]) & (dfd["window"] == comb["window"])
                 & (dfd["shift"] == comb["shift"]) & (dfd["initSubseq"] == comb["initSubseq"]) & (dfd["features"] == comb["features"])
                & (dfd["dataset"] == datasettemp) & (dfd["normalize"]== comb["normalize"])]
                f1list.append(row.iloc[-1][targetfield])
                #print(row)
            returnList.append(f1list)
            score_per_combination.append(statistics.median(f1list))
        except Exception as e:
            # print(row)
            # print(comb)
            # print(e)
            returnList.append(f1list)
            score_per_combination.append(0)


    return combinations[score_per_combination.index(max(score_per_combination))],max(score_per_combination),returnList[score_per_combination.index(max(score_per_combination))]

if __name__ == "__main__" :
    dfdyn = pd.read_csv("results.csv",header=0)
    dfstatic = pd.read_csv("resultsKR.csv",header=0)

    normlist=[True if kati else False for kati in dfdyn["features"]]
    dfdyn["normalize"]=normlist
    dfdyn["initSubseq"] = dfdyn["initSubseq"].fillna(value=-1)

    normlist = [True if kati else False for kati in dfstatic["features"]]
    dfstatic["normalize"] = normlist
    dfstatic["initSubseq"] = dfstatic["initSubseq"].fillna(value=-1)

    #f1Ex+Rpr,f1R+R
    #YahooA3Benchmark-TS
    #YahooA4Benchmark-TS
    #Yahoo_A1real_
    #Yahoo_A2synthetic_
    #print(get_best_on_dataset(dfdyn,"Yahoo_A2synthetic_",parmsspecific={},targetfield="f1Ex+Rpr"))

    #target="f1Ex+Rpr"
    target="f1R+R"
    #["k","r","window","shift","initSubseq","features","normalize"]
    print(get_best_on_dataset(dfdyn,"Yahoo_A2synthetic_",parmsspecific={"window":[200],"shift":[100]},targetfield=target))

    print(get_best_on_dataset(dfstatic,"Yahoo_A2synthetic_",parmsspecific={"window":[200],"shift":[100]},targetfield=target))

    print(get_best_on_all_datasets(dfdyn,parmsspecific={},targetfield=target))
    print(get_best_on_all_datasets(dfstatic,parmsspecific={},targetfield=target))




