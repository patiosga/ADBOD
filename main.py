import statistics

from tqdm import tqdm
from ExperimentTSB import experiment




fstatic =open("./results/resultsKR.csv", "a+")
fdyn =open("./results/results.csv", "a+")

def rumsingleexpkr(parms):
    #(k,r,window,slide,slidingWindow,features,normalize,i,dataset)
    k = parms[0]
    r = parms[1]
    window = parms[2]
    slide=parms[3]
    slidingWindow=parms[4]
    features=parms[5]
    normalize=parms[6]
    i = parms[7]
    dataset = parms[8]

    try:
        datasetname = f'{dataset}{i}_data.out'
        exp = experiment(slide=slide, window=window, k=k, r=r, filepath=datasetname, slidingWindow=slidingWindow,
                         onlystatic=True, features=features, normalize=normalize)
        if LOGRESULTS:
            fstatic.write(exp.static_to_write)
        return exp.staticf1
    except Exception as e:
        #print(e)
        return None


# Run Distance based outlier detection for all data files in dataset
# using given parameters, for multiple k and R parameters
def check_many_kt(slide=100, window=200,slidingWindow=None,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_"):
    maxall=0
    keepmax=None
    krparams=[]
    for k in [5, 10, 15, 20,30,40]:
        for r in [0.01,0.1, 0.5, 1, 2,5,10]:
            krparams.append((k,r))

    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(4)

    for qq in tqdm(range(len(krparams))):
        k=krparams[qq][0]
        r=krparams[qq][1]
        params=[]
        for i in range(100):
            params.append((k,r,window,slide,slidingWindow,features,normalize,i,dataset))
        results = pool.map(rumsingleexpkr, params)
        statallf1=[res for res in results if res is not None]
        if len(statallf1)==0:
            continue
        if maxall<sum(statallf1)/len(statallf1):
            maxall=sum(statallf1)/len(statallf1)
            keepmax=(k,r)
            print(maxall)
    print(maxall)
    print(keepmax)
    return keepmax

def onerun(slide=100, window=200,slidingWindow=None,keepmax=(10,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_"):
    dynallf1 = []
    statallf1 = []
    counter = 0

    for i in tqdm(range(100)):
        try:

            datasetname = f'{dataset}{i}_data.out'
            exp = experiment(slide=slide, window=window, k=keepmax[0], r=keepmax[1], filepath=datasetname, slidingWindow=slidingWindow,debug=debug,features=features,normalize=normalize)

            if LOGRESULTS:
                fstatic.write(exp.static_to_write)
                fdyn.write(exp.dyn_to_write)

            dynallf1.append(exp.dynamicf1)
            statallf1.append(exp.staticf1)
            counter += 1
            # print(f"static: {exp.staticf1}")
            # print(f"dyn: {exp.dynamicf1}")
        except Exception as e:
            #print(str(e))
            continue

    print(f" dyn mean: {sum(dynallf1) / counter}")
    print(f" dyn median: {statistics.median(dynallf1)}")

    print(f" static mean: {sum(statallf1) / counter}")
    print(f" static median: {statistics.median(statallf1)}")

    print(dynallf1)
    print(statallf1)


# Run technique to all data under the folder of given dataset using given parameters
def testYahoo(slide=100, window=200,slidingWindow=None,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A1real_"):
    # datasetname = f'./data/YAHOO/Yahoo_A1real_{i}_data.out'
    # datasetname = f'./data/YAHOO/YahooA3Benchmark-TS{i}_data.out'
    # datasetname = f'./data/YAHOO/YahooA4Benchmark-TS{i}_data.out'
    #"./data/YAHOO/Yahoo_A1synthetic_"

    keepmax=check_many_kt(slide=slide, window=window,slidingWindow=slidingWindow,features=features,normalize=normalize,dataset=dataset)
    onerun(slide=slide, window=window, slidingWindow=slidingWindow, keepmax=keepmax,features=features,normalize=normalize,dataset=dataset)



def run_all_for_dataset(window,slide,dataset):

    # no features no normalize
    #testYahoo(slide=slide, window=window, slidingWindow=2, features=False, normalize=False, dataset=dataset)
    #testYahoo(slide=slide, window=window, slidingWindow=None, features=False, normalize=False,dataset=dataset)
    #testYahoo(slide=slide, window=window, slidingWindow=10, features=False, normalize=False,dataset=dataset)

    # no features  normalize
    testYahoo(slide=slide, window=window, slidingWindow=2, features=False, normalize=True, dataset=dataset)
    testYahoo(slide=slide, window=window, slidingWindow=None, features=False, normalize=True,dataset=dataset)
    testYahoo(slide=slide, window=window, slidingWindow=10, features=False, normalize=True,dataset=dataset)

    # features
    #testYahoo(slide=slide, window=window, slidingWindow=None, features=True, normalize=True,dataset=dataset)
    #testYahoo(slide=slide, window=window, slidingWindow=10, features=True, normalize=True,dataset=dataset)

    #feature no normalize
    testYahoo(slide=slide, window=window, slidingWindow=None, features=True, normalize=False, dataset=dataset)
    testYahoo(slide=slide, window=window, slidingWindow=10, features=True, normalize=False,dataset=dataset)


LOGRESULTS=True
if __name__ == "__main__" :



    ######## For multiple combinations ###################################3
    run_all_for_dataset(200, 100, "./data/YAHOO/Yahoo_A1real_")
    run_all_for_dataset(200, 100, "./data/YAHOO/Yahoo_A2synthetic_")
    run_all_for_dataset(200, 100, "./data/YAHOO/YahooA3Benchmark-TS")
    run_all_for_dataset(200, 100, "./data/YAHOO/YahooA4Benchmark-TS")


    #### FOR a single combination ####################################3
    # testYahoo(slide=200, window=400, slidingWindow=10,features=True,normalize=False,dataset="./data/YAHOO/Yahoo_A1synthetic_")


    ### FOR Multuple KR using Distance based on all data files in dataset:

    # check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,
    #               dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/YahooA3Benchmark-TS")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/YahooA4Benchmark-TS")
    #check_many_kt(slide=200, window=400, slidingWindow=10, features=True, normalize=True,dataset="./data/YAHOO/Yahoo_A1real_")

    ### FOR a single parametrization of DynamicKR and Distance based K-R on all data files in dataset:
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(40,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(40,1.0),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")
    #onerun(slide=100, window=200,slidingWindow=10,keepmax=(15,0.5),debug=False,features=False,normalize=False,dataset="./data/YAHOO/Yahoo_A2synthetic_")

    # A single Experiment for a single data file using Distance based and Dynamic technique:
    #experiment(slide=slide, window=window, k=k, r=r, filepath=datasetname, slidingWindow=slidingWindow,
    #           onlystatic=True, features=features, normalize=normalize)

# Yahoo Synthetic
#           window    400, 600,
# Dyn (slide=2)      : 0.5, 0.66
# Static (slide=None): 0.26


# With None
#0.3265894308241438
#0.2659061895082558
#dyb
# [0.562962962962963, 0.32388663967611336, 0.37499999999999994, 0.19548872180451127, 0.4842105263157895, 0.45161290322580644, 0.13930348258706465, 0.1335559265442404, 0.17533129459734964, 0.5, 0.0, 0.31343283582089554, 0.676923076923077, 0.49019607843137253, 0.6804123711340206, 0.1023890784982935, 0.0332409972299169, 0.6575342465753424, 0.0574712643678161, 0.22641509433962265, 0.2857142857142857, 0.09090909090909091, 0.31178707224334595, 0.4752475247524753, 0.496551724137931, 0.0, 0.366412213740458, 0.08832807570977917, 0.5126353790613718, 0.5142857142857143, 0.031746031746031744, 0.0, 0.6942148760330578, 0.30222222222222217, 0.4246575342465754, 0.0, 0.27807486631016043, 0.09073359073359072, 0.0, 0.7, 0.4974619289340102, 0.4651162790697675, 0.3680981595092025, 0.4089456869009584, 0.16901408450704222, 0.6000000000000001, 0.0, 0.6575342465753424, 0.4067796610169491, 0.4973544973544974, 0.5142857142857142, 0.570281124497992, 0.3865546218487395, 0.33744855967078186, 0.5622775800711743, 0.0, 0.45901639344262296, 0.10267379679144384, 0.5692307692307692, 0.3114754098360656, 0.0, 0.3586206896551724, 0.0, 0.4166666666666667]
#KR 10,0.5 (best)
# [0.5103857566765578, 0.18777567737870193, 0.10162601626016259, 0.07837837837837837, 0.06199460916442049, 0.024725274725274724, 0.23244251087632073, 0.5637651821862348, 0.27718960538979787, 0.04884667571234735, 0.08649367930805056, 0.27163461538461536, 0.16464582003828973, 0.06720430107526881, 0.16347381864623242, 0.3199079401611047, 0.38295454545454544, 0.6, 0.38295454545454544, 0.2832167832167832, 0.3137254901960785, 0.1142857142857143, 0.7454068241469818, 0.16467630421118795, 0.14229765013054832, 0.1590763309813983, 0.12368421052631579, 0.3106682297772568, 0.2363747703612982, 0.2956521739130435, 0.12614980289093297, 0.8555708390646493, 0.10415293342122611, 0.23623995052566482, 0.125, 0.433879781420765, 0.18690400508582328, 0.8177299088649544, 0.21401752190237797, 0.09054593874833555, 0.10782380013149244, 0.5982905982905983, 0.1669805398618958, 0.2950236966824644, 0.2916419679905157, 0.19607843137254902, 0.3169590643274854, 0.6, 0.04087193460490463, 0.18112244897959184, 0.2398523985239852, 0.21733821733821734, 0.22966507177033493, 0.23405572755417958, 0.23405572755417958, 0.17836812144212524, 0.12285336856010569, 0.7544757033248082, 0.1213307240704501, 0.6516853932584269, 0.1681528662420382, 0.1543450064850843, 0.24212476837554048, 0.06924643584521385]



# 0.14 - 0.06 στο προτεινωμενο
# 0.17 - 0.19 στο 1