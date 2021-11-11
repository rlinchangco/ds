#!/usr/bin/python -tt

import sys, getopt, os, collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def globIt(crawlDir,extensions=[]):
    """
    stdlib version of a directory crawler searching for files with specific extensions
    """
    import glob
    list_of_files = []
    for extension in extensions:
        filePath = '{}*{}'.format(crawlDir,extension)
        print(f"Files to be parsed from:\n{filePath}")
        list_of_files += glob.glob(filePath)
    #latest_file = max(list_of_files, key=os.path.getctime)
    return list_of_files#, latest_file


def multi_parse(data_file, all_string=None, header_row=0):
    """
    Parses data file (accepts xlsx,tsv,csv) WITH header as first row.
    # read in chunks at a time(returns iterable if chunksize > 0)
    for df_chunk in pd.read_csv('data.csv', index_col=0, chunksize=8):
        print(df_chunk, end='\n\n')
    """
    df = pd.DataFrame()
    # reads csv, txt(tsv), and xlsx files
    if data_file.endswith('.csv'):
        if all_string:
            df = pd.read_csv(data_file, header=0, dtype=str)
        else:
            df = pd.read_csv(data_file, header=0)
    elif data_file.endswith('.tsv') or data_file.endswith('.txt'):
        if all_string:
            df = pd.read_csv(data_file, delimiter='\t', header=0, dtype=str)
        else:
            df = pd.read_csv(data_file, delimiter='\t', header=0)
    elif data_file.endswith('.xlsx'):
        if all_string:
            df = pd.read_excel(data_file, header=header_row, dtype=str)
        else:
            df = pd.read_excel(data_file, header=header_row)
    else:
        print(data_file)
        print(f"\n\nUnsupported file format\n\nPlease reformat...{data_file}")
        sys.exit()
    # add special characters to remove
    #extras = ['/', '\\', '']
    # df.columns = [sanitize_string(col_name, extras) for col_name in list(
    #    df.columns)]    # sanitize header string
    return df


def sanitize_string(a_string,swapspace='no',lower='no',extras=[]):
    """
    Removes special characters,explicitly specify lowercase,add others in extras list arg
    """
    if swapspace == 'yes':                              # replace space with underscore
        a_string.replace(' ','_')        
    to_remove = set(['"','.',' '])
    if len(extras) != 0:                                # add extra special characters for removal
        to_remove.update(extras)
    for special_char in to_remove:
        a_string = a_string.replace(special_char,'')
        if lower == 'yes':                              # lowercase entire string
            a_string.lower()
    return a_string


def check_nulls(df):
    """
    Checks for null values in a dataframe
    """
    if df.isnull().values.any():
        return f"Missing Data per column:\n{df.isnull().sum()}\nTotal Missing Data:{df.isnull().sum().sum()}"
    else:
        return "No missing data in dataset"


def collapse_multicolumns(df, new_column, cols):
    """
    Collapse multi-column codes into single columns
    Remove nulls
    new_column = str
    cols = list of column labels
    """
    df[new_column] = df[cols].values.tolist()
    df[new_column] = df[new_column].apply(lambda x:[el for el in x if not pd.isnull(el)])


def plot_histo(x,col,outPath=None):
    """
    """
    fig = plt.figure(figsize=(17,17))
    plt.hist(x,bins=50)
    plt.title(f"{col} Histogram")
    plt.xlabel(f"{col}")
    plt.ylabel('Frequency')
    if outPath:
        figure_file = "%s%s.png" % (outPath,col)
        fig.savefig(figure_file,dpi=fig.dpi)
    # else:
    #     plt.show() 
    # plt.close('all')    


def plotly_plot(df,pType,x,labels=None,title=None,y=None,outPath=None,color=None):
    """
    df = dataframe (pandas)
    pTypes: hist,bar,
    x = column label
    y = column label
    """
    if pType == 'hist':
        fig = px.histogram(df, x=x, title=title, labels=labels)
    elif pType == 'bar':
        fig = px.bar(df, x=x, y=y, title=title, labels=labels)
    elif pType == 'box':
        fig = px.box(df, y=x, title=title, labels=labels)
    elif pType == 'violin':
        fig = px.violin(df, y=x, box=True, points='all', title=title)
        #fig = px.violin(df, y=x, color=df.index, violinmode='overlay')
    elif pType == 'scatter':
        if color:
            fig = px.scatter(df, x=x, y=y, title=title, color=color)
        else:
            fig = px.scatter(df, x=x, y=y, title=title)
    ### Output
    if outPath:
        fig.write_html(f"{outPath}{x}.html")
    else:
        return fig


def find_optimal_k(xy):
    """
    Unsupervised, runs elbow method to determine optimum cluster number, checks via silhouette method
    Estimate cluster number by variance reduction
    """
    wcss = []               # within cluster sum of squares
    for i in range(1, 5):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(xy)
        wcss.append(kmeans.inertia_)
    previous_slope = 0
    # extract index(cluster num) when slope_dif drops below -1
    elbow = 1000
    stop = 0
    opt_elbow = list()
    for ind,val in enumerate(wcss):
        if ind != 0:
            slope = (val - wcss[ind-1])         # y(n) - y(n-1) / 1
            if slope == 0:
                opt_elbow = [ind,elbow]     # list(cluster num,slope_dif)
                stop = 1
                continue
            slope_ratio = previous_slope/slope
            slope_dif = previous_slope - slope
            if slope_dif > 0:
                slope_dif = -slope_dif
                elbow = slope_dif
            #print(elbow,slope_dif)
            if elbow < slope_dif:
                #print(elbow,slope_dif)
                elbow = slope_dif
                if elbow > -1 and stop == 0:
                    opt_elbow = [ind,elbow]     # list(cluster num,slope_dif)
                    stop = 1
            print("Cluster:{}\tSlope:{}\n".format(ind, slope_dif))
            previous_slope = slope
    #plot_validation(range(1, 11), wcss,["Elbow Method","Num clusters","WCSS"])    
    # Silhouette Validation
    # sil = []
    # kmax = 10
    # for k in range(2, kmax+1):  # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    #     current_k = KMeans(n_clusters = k, init = 'k-means++', random_state = 42).fit(xy)
    #     labels = current_k.labels_
    #     sil.append(silhouette_score(xy, labels, metric = 'euclidean'))
    # opt_clust1 = sil.index(max(sil)) + 2
    # opt_clust2 = sil.index(max(sil)) + 2
    # if opt_clust1 != opt_clust2:        # arbitrary assignment of 2nd best cluster
    #     opt_clust1 = opt_clust2
    # # add voting method?
    #plot_validation(range(2,kmax+1),sil,["Silhouette","num clusters","Silhouette Score"])
    #print("Elbow:\t{}\nSilhouette:\t{}".format(opt_elbow,opt_clust1))
    if len(opt_elbow)<1:
        print(f"Cannot find optimal K")
        return 2
    else:
        if opt_elbow[0] > 3:
            return 3
        else:
            return opt_elbow[0]


def lower_bic(bic_list):
    """
    Find best n via Bayesian IC
    """
    lowest_val = 0
    #second_low = 0
    best_n = 2
    for index,val in enumerate(bic_list):
        if val < lowest_val and index <= 3:
            lowest_val = val
            best_n = index+1
    
    return best_n


def main(argv):
    """
    Main method script
    """
    inputPath = ''
    outputFile = None
    fileEnd = 'csv'
    htmlList = []                                               # list of plotly figure visualizations for writing out

    ### CLI Argument definition and parsing
    try:
        opts, args = getopt.getopt(
            argv, "hi:e:o:", ["inputPath=", "fileEnd", "ofile="])
    except getopt.GetoptError:
        print('fraud.py -i <inputPath> -e <fileEnd> -o <ofile>\n\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fraud.py -i <inputPath> -e <fileEnd> -o <ofile>\n\n')
            print("-e is the file ending of files to parse, eg. csv,txt,xlsx")
            sys.exit()
        elif opt in ("-i", "--inputPath"):
            inputPath = arg
        elif opt in ("-e", "--fileEnd"):
            fileEnd = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg

    if not inputPath.endswith('/'):                             # add trailing directory separator if needed
        inputPath += '/'
    stdReport = open(inputPath+'runReport.txt','w')
    stdReport.write(f'Input path is {inputPath}\n')
    fileList = globIt(inputPath,[fileEnd])
    stdReport.write(f"Files retrieved:\n{fileList}\n")
    dataList = [(multi_parse(dataFile, all_string=True, header_row=0),dataFile.split('/')[-1]) for dataFile in fileList]
    benefList = []
    claimsDf = None
    for dframe in dataList:
        shape = dframe[0].shape
        stdReport.write(f"File {dframe[1]} has:\nshape: {shape}\ncolumns: {dframe[0].columns}\n")
        if shape[1] != 32:
            claimsDf = dframe[0]
        else:
            benefList.append(dframe[0])
    
    ### Beneficiary Dataframe
    benefDf = pd.concat(benefList)
    stdReport.write("Benefits Data:\n")
    stdReport.write(f"Rows,Columns: {benefDf.shape}\n")
    stdReport.write(f"Column Labels:\n{benefDf.columns}\n")
    stdReport.write("Beneficiary Null Report\n")
    stdReport.write(check_nulls(benefDf))    
    # print(f"BENEFICIARIES General Info:\n")
    # print(benefDf.head())
    
    ### Outpatient Claims Dataframe
    stdReport.write("Claims Data:\n")
    stdReport.write(f"Rows,Columns:{claimsDf.shape}\n")
    stdReport.write(f"Column Labels:\n{claimsDf.columns}\n")
    stdReport.write("Outpatient Claims Null Report\n")
    stdReport.write(check_nulls(claimsDf))
    # print(f"CLAIMS General Info:\n")
    # print(claimsDf.info())    

    ### Dataframe of only claims with office visit codes covering 99211-5 (scope of analysis)
    offVisList = []
    offCodesList = ['99211', '99212', '99213', '99214', '99215']
    officeCodes = '|'.join(offCodesList)
    for col in claimsDf.columns[31:]:
        offVisList.append(claimsDf.loc[claimsDf[col].str.contains(officeCodes, case=False, na=False)])
    offVisDf = pd.concat(offVisList)

    ### Count occurrence of each visit level across dataset
    offVisCounts = {}
    for offCode in offCodesList:
        total = 0
        for col in offVisDf.columns[31:]:
            total += offVisDf[col].str.contains(offCode).sum()
        if offCode not in offVisCounts:
            offVisCounts[offCode] = total
    offCounts = pd.DataFrame.from_dict(offVisCounts, orient='index')
    # visualize
    htmlList.append(plotly_plot(offCounts,'bar',0,title="Distribution of Office Visit Types",labels={'0':'Occurrence','index':'Office Visit HCPCS Codes'}))

    ### Convert date string to year and month columns
    offVisDf["CLM_FROM_DT_YEAR"] = offVisDf["CLM_FROM_DT"].astype(str).str.slice(0,4)               # format = "YYYY"
    offVisDf["CLM_FROM_DT_MONTH"] = offVisDf["CLM_FROM_DT"].astype(str).str.slice(0,6)              # format = "YYYYMM"

    ### Collapse multi-column codes into single columns
    collapseList = [('HCPCS_CD',claimsDf.columns[31:]),('ICD9_DGNS_CD',claimsDf.columns[12:22]),('ICD9_PRCDR_CD',claimsDf.columns[22:28])]
    for collapsePair in collapseList:
        collapse_multicolumns(offVisDf, collapsePair[0], collapsePair[1])
    
    ### Condensed offVisDf
    useVars = ["DESYNPUF_ID","CLM_ID","PRVDR_NUM",
                "CLM_FROM_DT_YEAR","CLM_FROM_DT_MONTH",
                "CLM_PMT_AMT","NCH_BENE_PTB_DDCTBL_AMT","NCH_BENE_PTB_COINSRNC_AMT","NCH_PRMRY_PYR_CLM_PD_AMT",
                "AT_PHYSN_NPI","OP_PHYSN_NPI","OT_PHYSN_NPI",
                "ADMTNG_ICD9_DGNS_CD","ICD9_DGNS_CD","ICD9_PRCDR_CD","HCPCS_CD"]
    cndsdOffVisDf = offVisDf[useVars]
    ### Convert data types for float
    typesDict = {"CLM_PMT_AMT":float,"NCH_BENE_PTB_DDCTBL_AMT":float,"NCH_BENE_PTB_COINSRNC_AMT":float,"NCH_PRMRY_PYR_CLM_PD_AMT":float}
    for col, col_type in typesDict.items():
        cndsdOffVisDf[col] = cndsdOffVisDf[col].astype(col_type)
    cndsdOffVisDf["NUM_DGNS"] = cndsdOffVisDf["ICD9_DGNS_CD"].str.len()
    cndsdOffVisDf["NUM_PRCDR"] = cndsdOffVisDf["ICD9_PRCDR_CD"].str.len()
    cndsdOffVisDf["NUM_HCPCS"] = cndsdOffVisDf["HCPCS_CD"].str.len()
    # print(cndsdOffVisDf[["NUM_DGNS","NUM_PRCDR","NUM_HCPCS"]].describe())
    ### Count occurrence of office visits per claim
    cndsdOffVisDf['HCPCS_CD_STR'] = [','.join(map(str, l)) for l in cndsdOffVisDf['HCPCS_CD']]                  # new column collapsing list column into comma sep string column
    for offCode in offCodesList:
        cndsdOffVisDf[f"{offCode}_COUNT"] = cndsdOffVisDf['HCPCS_CD_STR'].astype(str).str.count(offCode)        
    cndsdOffVisDf['ALL_VIS_COUNT'] = cndsdOffVisDf['HCPCS_CD_STR'].astype(str).str.count(officeCodes)
    stdReport.write(f"Rows,Columns: {cndsdOffVisDf.shape}\n")
    # print(f"General Info:\n{offVisDf.info()}\n")
    stdReport.write(f"Inspect Outpatient Claim Values:\n{cndsdOffVisDf.head()}\n")
    ### Investigate top 10 number of members, providers, physicians, and claim amounts
    stdReport.write(f"Inspect top ten for claims:\n")
    for col in ['DESYNPUF_ID','PRVDR_NUM','AT_PHYSN_NPI','OP_PHYSN_NPI','OT_PHYSN_NPI','CLM_PMT_AMT']:
        stdReport.write(f"by {col}:\n{cndsdOffVisDf[col].value_counts()[:10]}\n")
    
    #### DATA INSPECTION 
    ### Investigate NA CLAIM DATES
    noDateDf = cndsdOffVisDf.loc[offVisDf["CLM_FROM_DT"].isna()]
    stdReport.write(f"Claims missing dates:\n{noDateDf.describe()}\n")
    # High potential for fraud as no Office visit types are all 99211, with no procedures, and highly variant claim amounts
    for col in ['DESYNPUF_ID','PRVDR_NUM','AT_PHYSN_NPI','OP_PHYSN_NPI','OT_PHYSN_NPI','CLM_PMT_AMT']:
        print(noDateDf[col].value_counts())
    stdReport.write(f"Descriptive statistics of missing date claims costs:\n{noDateDf['CLM_PMT_AMT'].describe()}\n")
    # count      88.000000
    # mean      609.772727
    # std       934.060444
    # min         0.000000
    # 25%        60.000000
    # 50%       100.000000
    # 75%       700.000000
    # max      3300.000000
    stdReport.write(f"Looking at physicians for missid date claims:\n{noDateDf[['AT_PHYSN_NPI','OP_PHYSN_NPI','OT_PHYSN_NPI']]}\n")
    # All are null when dates are null
    stdReport.write(f"Percent fraudulent claims from missing data: {noDateDf.shape[0]/cndsdOffVisDf.shape[0]*100}%\n")

    #### FEATURE ENGINEERING    
    ### CLAIMS
    visPerYear = cndsdOffVisDf.groupby(["CLM_FROM_DT_YEAR"])["ALL_VIS_COUNT"].agg('sum')
    htmlList.append(plotly_plot(visPerYear,"bar","ALL_VIS_COUNT",title="Overall Office Visits Per Year"))
    visPerMonth = cndsdOffVisDf.groupby(["CLM_FROM_DT_MONTH"])["ALL_VIS_COUNT"].agg('sum')
    htmlList.append(plotly_plot(visPerMonth,"bar","ALL_VIS_COUNT",title="Overall Office Visits Per Month"))
    visitsPMPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])["ALL_VIS_COUNT"].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').DESYNPUF_ID.nunique()
    htmlList.append(plotly_plot(visitsPMPM,"bar",0,title="Visits Per Member Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Visits per Member'}))
    visitsPPPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])["ALL_VIS_COUNT"].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').PRVDR_NUM.nunique()
    htmlList.append(plotly_plot(visitsPPPM,"bar",0,title="Visits Per Provider Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Visits per Provider'}))
    visitsPDPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])["ALL_VIS_COUNT"].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').AT_PHYSN_NPI.nunique()
    htmlList.append(plotly_plot(visitsPDPM,"bar",0,title="Visits Per Physician Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Visits per Physician'}))
    # yearCounts = cndsdOffVisDf["CLM_FROM_DT_YEAR"].value_counts()                                               # assumption that each row has unique claim_id
    # monthCounts = cndsdOffVisDf["CLM_FROM_DT_MONTH"].value_counts()                                             # assumption that each row has unique claim_id
    # htmlList.append(plotly_plot(yearCounts,"bar","CLM_FROM_DT_YEAR",title="Claims Per Year"))
    # htmlList.append(plotly_plot(monthCounts,"bar","CLM_FROM_DT_MONTH",title="Claims Per Month"))

    #### COSTS
    ### Overall distribution of claim costs
    htmlList.append(plotly_plot(cndsdOffVisDf,"hist","CLM_PMT_AMT",title="Distribution of Claim Costs"))
    ### Outlier search of claim costs
    htmlList.append(plotly_plot(cndsdOffVisDf,"violin","CLM_PMT_AMT",title="Boxplot/Distribution of Claims Costs"))
    ### Claim costs over months
    clmPerMonth = cndsdOffVisDf.groupby(["CLM_FROM_DT_MONTH"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmPerMonth,"bar","CLM_PMT_AMT",title="Claim Costs Per Month",labels={"CLM_PMT_AMT": 'Summed Claim Costs', "CLM_FROM_DT_MONTH": "Month"}))
    ### Claim costs over years
    clmPerYear = cndsdOffVisDf.groupby(["CLM_FROM_DT_YEAR"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmPerYear,"bar","CLM_PMT_AMT",title="Claim Costs Per Year",labels={"CLM_PMT_AMT": 'Summed Claim Costs', "CLM_FROM_DT_YEAR": "YEAR"}))
    ### Claim costs per member per month
    costPMPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').DESYNPUF_ID.nunique()
    sortedCostPMPM = costPMPM.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedCostPMPM,"bar",0,title="Claim Costs Per Member Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Member Claim Cost($)'}))
    ### Claim costs per provider per month
    costPPPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').PRVDR_NUM.nunique()
    sortedCostPPPM = costPPPM.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedCostPPPM,"bar",0,title="Claim Costs Per Provider Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Provider Claim Cost($)'}))
    ### Claim costs per member per year
    costPMPY=cndsdOffVisDf.groupby(['CLM_FROM_DT_YEAR'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_YEAR').DESYNPUF_ID.nunique()
    sortedCostPMPY = costPMPY.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedCostPMPY,"bar",0,title="Claim Costs Per Member Per Year",labels={'CLM_FROM_DT_YEAR':'Year','0':'Average Member Claim Cost($)'}))
    ### Claim costs per provider per year
    costPPPY=cndsdOffVisDf.groupby(['CLM_FROM_DT_YEAR'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_YEAR').PRVDR_NUM.nunique()
    sortedCostPPPY = costPPPY.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedCostPPPY,"bar",0,title="Claim Costs Per Provider Per Year",labels={'CLM_FROM_DT_YEAR':'Year','0':'Average Provider Claim Cost($)'}))
    ### Claim costs for each member
    clmCostByMember = offVisDf.groupby(["DESYNPUF_ID"])["CLM_PMT_AMT"].agg('sum')
    sortedClmCostByMember = clmCostByMember.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedClmCostByMember,"bar","CLM_PMT_AMT",title="Claim Costs Per Provider"))
    ### Claim costs for each provider
    clmCostByProvider = cndsdOffVisDf.groupby(["PRVDR_NUM"])["CLM_PMT_AMT"].agg('sum')
    sortedClmCostByProvider = clmCostByProvider.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedClmCostByProvider,"bar","CLM_PMT_AMT",title="Claim Costs Per Provider"))
    ### Claim costs for each attending physician
    clmCostByPhysician = cndsdOffVisDf.groupby(["AT_PHYSN_NPI"])["CLM_PMT_AMT"].agg('sum')
    sortedClmCostByPhysician = clmCostByPhysician.sort_values(ascending=False)[:20]
    htmlList.append(plotly_plot(sortedClmCostByPhysician,"bar","CLM_PMT_AMT",title="Claim Costs Per Physician"))
    
    # #### PCA
    # features = []                                       # TBD
    # x = cndsdOffVisDf[features]
    # x = StandardScaler().fit_transform(x)       
    # pca = PCA(n_components=2)
    # # pca = PCA(.95)                                    # or find optimal components by capturing 95% variance
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ["principal component 1", "principal component 2"])
    
    # ### Plot PCA and write to html
    # htmlList.append(plotly_plot(principalDf,"scatter","principal component 1", y="principal component 2",title="PCA"))
    
    # #### Clustering
    # train = []                                          # TBD
    # test = []                                           # TBD
    # ### KMeans
    ## opt_k = find_optimal_k(principalDf)                   # If n is unknown, use as argument for n_clusters
    # k_fit = KMeans(n_clusters=2, init = 'k-means++', random_state=42,).fit(train) # Assume best n is 2 for finding fraud and not fraud
    # classifications = k_fit.predict(test)
    # ### Gaussian Mixture Models                             
    ## n_components = np.arange(1, 6)
    ## models = [GaussianMixture(n, covariance_type='full', random_state=0, warm_start=True).fit(train) for n in n_components]
    ## bic_list = [m.bic(train) for m in models]                          # run BIC once
    ## best_n = lower_bic(bic_list)                           # If n is unknown, use as argument for n_components
    # gmm = GaussianMixture(n_components=2, covariance_type='full', warm_start=True).fit(train) # Assume best n is 2 for finding fraud and not fraud
    # gmm_probabilities = gmm.predict_proba(test)             # Probability for each position to be in each cluster
    # clust_cent = gmm.cluster_centers_                       # cluster centers
    # gmm_pred = gmm.predict(test)                            # prediction of clusters

    # ### Plot Clusters to html
    # htmlList.append(plotly_plot(principalDf,"scatter","principal component 1",y="principal component 2",title="Fraudulent Claims KMeans",color=classifications))
    # htmlList.append(plotly_plot(principalDf,"scatter","principal component 1",y="principal component 2",title="Fraudulent Claims GMM",color=gmm_pred))
    
    ### Write out plotly visuzlizations to HTML file for interactive
    with open(f"{inputPath}fraud_visualizations.html", 'a') as f:
        for plotFig in htmlList:
            f.write(plotFig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()

    ### End analysis
    stdReport.flush()
    stdReport.close()

if __name__ == "__main__":
    main(sys.argv[1:])