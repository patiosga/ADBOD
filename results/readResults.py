import math
import statistics

import pandas as pd
from autorank import autorank, plot_stats, create_report


def get_best_on_all_datasets(dfd,parmsspecific={},targetfield="f1Ex+Rpr"):

    alldatasets = dfd["dataset"].unique()

    #temp = alldatasets[0]
    # method,k,r,window,shift,initSubseq,ActSubSeq,Existence,Rrecall,Rprecision,f1Ex+Rpr,f1R+R,features,dataset
    # combinations = k,r,window,shift,initSubseq,features
    Parametercolumns = ["k", "r", "window", "shift", "initSubseq", "features", "normalize"]
    ParamValues = []
    combinations = []
    for paramm in parmsspecific.keys():
        dfd = dfd[dfd[paramm].isin(parmsspecific[paramm])]
    for param in Parametercolumns:
        if param in parmsspecific.keys():
            ParamValues.append(parmsspecific[param])
        else:
            ParamValues.append([vl for vl in dfd[param].unique()])

    for k in ParamValues[0]:
        for r in ParamValues[1]:
            for window in ParamValues[2]:
                for initSubseq in ParamValues[4]:
                    for features in ParamValues[5]:
                        for normalize in ParamValues[6]:
                            combinations.append(
                                {"k": k, "r": r, "window": window, "shift": window // 2, "initSubseq": initSubseq,
                                 "features": features, "normalize": normalize})
    print(f"combinations: {len(combinations)}")
    # for each combination calculate median ?
    score_per_combination = []

    params = []
    for comb in combinations:
        # print()
        params.append((dfd,comb,alldatasets,targetfield))

    returnList = pool.map(evaluate_combination, params)

    for enum,f1list in enumerate(returnList):
        try:
            score_per_combination.append(statistics.median(f1list))
        except:
            score_per_combination.append(-1)
            #print(f"Exception, combination:{combinations[enum]}")
    return alldatasets,combinations[score_per_combination.index(max(score_per_combination))],max(score_per_combination),returnList[score_per_combination.index(max(score_per_combination))]

def evaluate_combination(paramset):
    dfd=paramset[0]
    comb=paramset[1]
    alldatasets=paramset[2]
    targetfield=paramset[3]

    try:
        f1list = []
        #for datasettemp in alldatasets:
        row = dfd.loc[(dfd["k"] == comb["k"]) & (dfd["r"] == comb["r"]) & (dfd["window"] == comb["window"])
                          & (dfd["shift"] == comb["shift"]) & (dfd["initSubseq"] == comb["initSubseq"]) & (
                                  dfd["features"] == comb["features"])
                           & (dfd["normalize"] == comb["normalize"]) & (dfd["dataset"].isin(alldatasets))]


        #(dfd["dataset"] == datasettemp)
        f1temp = row[targetfield].values
        if len(f1temp)==0:
            return []
        if abs(len(f1temp)-len(alldatasets))>10:
            print(f"{comb} -> {len(f1temp)} != {len(alldatasets)}")

        f1list=[kati for kati in f1temp]

    except Exception as e:
        print("Esception")
        print(e)
    return f1list

def get_best_on_dataset(dfd,dataset,parmsspecific={},targetfield="f1Ex+Rpr"):
    dfd=dfd[dfd["dataset"].str.contains(dataset)]

    alldatasets=dfd["dataset"].unique()

    temp=alldatasets[0]
    # method,k,r,window,shift,initSubseq,ActSubSeq,Existence,Rrecall,Rprecision,f1Ex+Rpr,f1R+R,features,dataset
    # combinations = k,r,window,shift,initSubseq,features
    Parametercolumns=["k","r","window","shift","initSubseq","features","normalize"]
    ParamValues=[]
    combinations=[]
    for paramm  in parmsspecific.keys():
        dfd=dfd[dfd[paramm].isin(parmsspecific[paramm])]
    for param in Parametercolumns:
        if param in parmsspecific.keys():
            ParamValues.append(parmsspecific[param])
        else:
            ParamValues.append([vl for vl in dfd[param].unique()])

    for k in ParamValues[0]:
        for r in ParamValues[1]:
            for window in ParamValues[2]:
                for initSubseq in ParamValues[4]:
                    for features in ParamValues[5]:
                        for normalize in ParamValues[6]:
                            combinations.append({"k":k,"r":r,"window":window,"shift":window//2,"initSubseq":initSubseq,"features":features,"normalize":normalize})
    print(f"combinations: {len(combinations)}")
    # for each combination calculate median ?
    score_per_combination = []

    params = []
    for comb in combinations:
        # print()
        params.append((dfd, comb, alldatasets, targetfield))

    returnList = pool.map(evaluate_combination, params)

    for enum, f1list in enumerate(returnList):
        try:
            score_per_combination.append(statistics.median(f1list))
        except:
            score_per_combination.append(-1)
            #print(f"Exception, combination:{combinations[enum]}")

    return alldatasets,combinations[score_per_combination.index(max(score_per_combination))],max(score_per_combination),returnList[score_per_combination.index(max(score_per_combination))]




from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(8)

def plotgraph(static,staticnames, dyn,dynNames,target):
    if "Ex" in target:
        level=1
        atag="a"
        btag="b"
        ctag="c"
    else:
        atag = "d"
        btag = "e"
        ctag = "f"
        level=2
    import matplotlib.pyplot as plt

    fstatic = []
    fdyn = []
    fstatic1=[]
    fdyn1=[]
    fstatic2 = []
    fdyn2 = []
    fstatic3 = []
    fdyn3 = []
    fstatic4 = []
    fdyn4 = []
    marker=[]
    for sc,dname in zip(static,staticnames):
        if dname in dynNames:
            fstatic.append(sc)
            fdyn.append(dyn[dynNames.index(dname)])
            if "A1real" in dname:
                fstatic1.append(sc)
                fdyn1.append(dyn[dynNames.index(dname)])
            elif "A2synth" in dname:
                fstatic2.append(sc)
                fdyn2.append(dyn[dynNames.index(dname)])
            elif "A3" in dname:
                fstatic3.append(sc)
                fdyn3.append(dyn[dynNames.index(dname)])
            else:
                fstatic4.append(sc)
                fdyn4.append(dyn[dynNames.index(dname)])
    # dict = {'Stat': fstatic,'Dyn': fdyn}
    #  #       'A2dyn': fdyn2, 'A2stat': fstatic2,
    #  #       'A3dyn': fdyn3, 'A3stat': fstatic3,
    #  #       'A4dyn': fdyn4, 'A4stat': fstatic4}
    #
    # data = pd.DataFrame(dict)
    # results=autorank(data,force_mode="parametric")
    # create_report(results)
    # plot_stats(results)
    # plt.show()
    plt.subplot(221)
    plt.gca().set_title(f"({atag})")
    boxes=plt.boxplot([fdyn1,fstatic1,fdyn2,fstatic2,fdyn3,fstatic3,fdyn4,fstatic4],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=["Dyn(A1)","DOD(A1)","Dyn(A2)","DOD(A2)","Dyn(A3)","DOD(A3)","Dyn(A4)","DOD(A4)"])  # will be used to label x-ticks

    for i,box in enumerate(boxes['boxes']):
        # change outline color
        if i%2==0:
            box.set(color='black')
            # change fill color
            box.set(facecolor='blue')
            # change hatch
            box.set(hatch='/')
        else:
            box.set(color='black')
            # change fill color
            box.set(facecolor='green')
    plt.ylabel(f"AD{level} F1")
    plt.subplot(223)
    plt.gca().set_title(f"({btag})")
    boxes=plt.boxplot([fdyn,fstatic],
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=["Dyn(All)", "DOD(All)"])

    for i,box in enumerate(boxes['boxes']):
        # change outline color
        if i%2==0:
            box.set(color='black')
            # change fill color
            box.set(facecolor='blue')
            # change hatch
            box.set(hatch='/')
        else:
            box.set(color='black')
            # change fill color
            box.set(facecolor='green')
    plt.ylabel(f"AD{level} F1")
    plt.subplot(122)
    plt.gca().set_title(f"({ctag})")
    plt.scatter(fstatic1, fdyn1,marker="*",color="orange")
    plt.scatter(fstatic2, fdyn2,marker="s",color="blue")
    plt.scatter(fstatic3, fdyn3,marker="o",color="magenta")
    plt.scatter(fstatic4, fdyn4,marker="D",color="red")
    plt.plot([0,1], [0,1],color="green",linewidth=2)

    plt.xlabel(f"DOD(AD{level} F1)")
    plt.ylabel(f"Dyn(AD{level} F1)")

    plt.xlim([-0.05, 1.1])
    plt.ylim([-0.05, 1.1])
    plt.show()

def constantKR_against_Dyn():
    dfdyn = pd.read_csv("results_half.csv", header=0)
    dfstatic = pd.read_csv("resultsKR_half.csv", header=0)

    dfdyn["initSubseq"] = dfdyn["initSubseq"].fillna(value=-1)

    dfstatic["initSubseq"] = dfstatic["initSubseq"].fillna(value=-1)

    # f1Ex+Rpr,f1R+R
    # YahooA3Benchmark-TS
    # YahooA4Benchmark-TS
    # Yahoo_A1real_
    # Yahoo_A2synthetic_
    # print(get_best_on_dataset(dfdyn,"Yahoo_A2synthetic_",parmsspecific={},targetfield="f1Ex+Rpr"))

    target = "f1Ex+Rpr"
    # target="f1R+R"
    # ["k","r","window","shift","initSubseq","features","normalize"]
    # combination,scoremedian,allscores=get_best_on_dataset(dfdyn,"Yahoo_A2synthetic_",parmsspecific={"features":[False],"normalize":[True]},targetfield=target)
    # print(f"score: {scoremedian}, len={len(allscores)}")
    # print(allscores)
    #
    # combination,scoremedian,allscores=get_best_on_dataset(dfstatic,"Yahoo_A2synthetic_",parmsspecific={"features":[False],"normalize":[True]},targetfield=target)
    # print(f"score: {scoremedian}, len={len(allscores)}")
    # print(allscores)
    # print(combination)

    ######################################33
    print("##########################################################################################333")
    alldatasets, combination, scoremedian, allscores = get_best_on_all_datasets(dfdyn,
                                                                                parmsspecific={"normalize": [False]},
                                                                                targetfield=target)
    print(f"score: {scoremedian}, len={len(allscores)}")
    print(allscores)

    dynNames = []
    dynscore = []
    for d, s in zip(alldatasets, allscores):
        if s != -1:
            dynNames.append(d)
            dynscore.append(s)

    alldatasets, combination, scoremedian, allscores = get_best_on_all_datasets(dfstatic,
                                                                                parmsspecific={"normalize": [False]},
                                                                                targetfield=target)
    print(f"score: {scoremedian}, len={len(allscores)}")
    print(allscores)
    print(combination)

    statNames = []
    statscore = []
    for d, s in zip(alldatasets, allscores):
        if s != -1:
            statNames.append(d)
            statscore.append(s)

    plotgraph(statscore, statNames, dynscore, dynNames, target)
    # print(get_best_on_all_datasets(dfdyn,parmsspecific={},targetfield=target))
    # print(get_best_on_all_datasets(dfstatic,parmsspecific={},targetfield=target))

    # USING HALF
    # I HAVE
    # Subsequence length: None, 10, 1,2
    # Windows: 400,200
    # Feature: True using normalize, False without Normalize

def get_best_with_different_k_R(dfd,parmsspecific={"normalize": [False]},targetfield="f1Ex+Rpr"):
    alldatasets = dfd["dataset"].unique()
    for paramm in parmsspecific.keys():
        dfd = dfd[dfd[paramm].isin(parmsspecific[paramm])]

    # method,k,r,window,shift,initSubseq,ActSubSeq,Existence,Rrecall,Rprecision,f1Ex+Rpr,f1R+R,features,dataset
    # combinations = k,r,window,shift,initSubseq,features
    Parametercolumns = ["window", "shift", "initSubseq", "features", "normalize"]
    ParamValues = []
    combinations = []

    for param in Parametercolumns:
        if param in parmsspecific.keys():
            ParamValues.append(parmsspecific[param])
        else:
            ParamValues.append([vl for vl in dfd[param].unique()])

    for window in ParamValues[0]:
        for initSubseq in ParamValues[2]:
            for features in ParamValues[3]:
                for normalize in ParamValues[4]:
                    combinations.append(
                        {"window": window, "shift": window // 2, "initSubseq": initSubseq,
                         "features": features, "normalize": normalize})
    print(f"combinations: {len(combinations)}")
    # for each combination calculate median ?
    score_per_combination = []
    datasetsallreturn = []

    params = []
    for comb in combinations:
        params.append((dfd, comb, alldatasets, targetfield))

    returnList = pool.map(evaluate_combination_best_kr, params)

    for enum, tupp in enumerate(returnList):
        score_per_combination.append(statistics.median(tupp[0]))
        datasetsallreturn.append(tupp[1])

    return datasetsallreturn[score_per_combination.index(max(score_per_combination))], combinations[score_per_combination.index(max(score_per_combination))], max(
        score_per_combination), returnList[score_per_combination.index(max(score_per_combination))][0]


def evaluate_combination_best_kr(paramset):
    dfd=paramset[0]
    comb=paramset[1]
    alldatasets=paramset[2]
    targetfield=paramset[3]


    f1list = []
    #for datasettemp in alldatasets:
    row = dfd.loc[(dfd["window"] == comb["window"]) & (dfd["shift"] == comb["shift"])
                  & (dfd["initSubseq"] == comb["initSubseq"]) & (dfd["features"] == comb["features"])
                       & (dfd["normalize"] == comb["normalize"]) & (dfd["dataset"].isin(alldatasets))]

    f1temp = []
    toreturndatasets=[]
    for dataset in alldatasets:
        try:
            max_target=row.loc[row["dataset"]==dataset][targetfield].max()
            if math.isnan(max_target):
                continue
            f1temp.append(max_target)
            toreturndatasets.append(dataset)
        except Exception as e:
            print("Esception")
            print(e)
            continue
    #(dfd["dataset"] == datasettemp)

    if len(f1temp)==0:
        return []
    if abs(len(f1temp)-len(alldatasets))>10:
        print(f"{comb} -> {len(f1temp)} != {len(alldatasets)}")

    f1list=[kati for kati in f1temp]

    return (f1list,toreturndatasets)


def PertimeseriesBest_vs_Dyn():
    dfdyn = pd.read_csv("results_half.csv", header=0)
    dfstatic = pd.read_csv("resultsKR_half.csv", header=0)

    dfdyn["initSubseq"] = dfdyn["initSubseq"].fillna(value=-1)

    dfstatic["initSubseq"] = dfstatic["initSubseq"].fillna(value=-1)

    target = "f1Ex+Rpr"
    #target="f1R+R"

    print("##########################################################################################333")
    alldatasets, combination, scoremedian, allscores = get_best_on_all_datasets(dfdyn,
                                                                                parmsspecific={"normalize": [False]},
                                                                                targetfield=target)
    print(f"score: {scoremedian}, len={len(allscores)}")
    print(allscores)

    dynNames = []
    dynscore = []
    for d, s in zip(alldatasets, allscores):
        if s != -1:
            dynNames.append(d)
            dynscore.append(s)

    alldatasets, combination, scoremedian, allscores = get_best_with_different_k_R(dfstatic,
                                                                                parmsspecific={"normalize": [False]},
                                                                                targetfield=target)
    print(f"score: {scoremedian}, len={len(allscores)}")
    print(allscores)
    print(combination)

    statNames = []
    statscore = []
    for d, s in zip(alldatasets, allscores):
        if s != -1:
            statNames.append(d)
            statscore.append(s)
            #print(s)
    plotgraph(statscore, statNames, dynscore, dynNames, target)





if __name__ == "__main__" :
    #constantKR_against_Dyn()
    PertimeseriesBest_vs_Dyn()



