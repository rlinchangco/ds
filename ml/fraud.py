#!/usr/bin/python -tt

import sys, getopt, os, collections
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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


def plotly_plot(df,pType,x,labels=None,title=None,y=None,outPath=None):
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
    ### Output
    if outPath:
        fig.write_html(f"{outPath}{x}.html")
    else:
        return fig


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
    print(cndsdOffVisDf[["NUM_DGNS","NUM_PRCDR","NUM_HCPCS"]].describe())
    ### Count occurrence of office visits per claim
    cndsdOffVisDf['HCPCS_CD_STR'] = [','.join(map(str, l)) for l in cndsdOffVisDf['HCPCS_CD']]                  # new column collapsing list column into comma sep string column
    for offCode in offCodesList:
        cndsdOffVisDf[f"{offCode}_COUNT"] = cndsdOffVisDf['HCPCS_CD_STR'].astype(str).str.count(offCode)        
    cndsdOffVisDf['ALL_VIS_COUNT'] = cndsdOffVisDf['HCPCS_CD_STR'].astype(str).str.count(officeCodes)
    stdReport.write(f"Rows,Columns: {cndsdOffVisDf.shape}\n")
    # print(f"General Info:\n{offVisDf.info()}\n")
    stdReport.write(f"Inspect Outpatient Claim Values:\n{cndsdOffVisDf.head()}\n")

    #### DATA INSPECTION 
    ### Investigate NA CLAIM DATES
    noDateDf = cndsdOffVisDf.loc[offVisDf["CLM_FROM_DT"].isna()]
    # print(noDateDf.describe())
    ## High potential for fraud as no Office visit types are all 99211, with no procedures, and highly variant claim amounts
    # for col in ['DESYNPUF_ID','PRVDR_NUM','AT_PHYSN_NPI','OP_PHYSN_NPI','OT_PHYSN_NPI','CLM_PMT_AMT']:
    #     print(noDateDf[col].value_counts())
    # print(noDateDf['CLM_PMT_AMT'].describe())
    ## count      88.000000
    ## mean      609.772727
    ## std       934.060444
    ## min         0.000000
    ## 25%        60.000000
    ## 50%       100.000000
    ## 75%       700.000000
    ## max      3300.000000
    # print(noDateDf[['AT_PHYSN_NPI','OP_PHYSN_NPI','OT_PHYSN_NPI']])
    ## All are null when dates are null
    print(f"Percent fraudulent claims from missing data: {noDateDf.shape[0]/cndsdOffVisDf.shape[0]*100}%")
    """
    #### FEATURE ENGINEERING    
    ### CLAIMS
    visPerYear = cndsdOffVisDf.groupby(["CLM_FROM_DT_YEAR"])["ALL_VIS_COUNT"].agg('sum')
    visPerMonth = cndsdOffVisDf.groupby(["CLM_FROM_DT_MONTH"])["ALL_VIS_COUNT"].agg('sum')
    htmlList.append(plotly_plot(visPerYear,"bar","ALL_VIS_COUNT",title="Overall Office Visits Per Year"))
    htmlList.append(plotly_plot(visPerMonth,"bar","ALL_VIS_COUNT",title="Overall Office Visits Per Month"))
    yearCounts = cndsdOffVisDf["CLM_FROM_DT_YEAR"].value_counts()                                               # assumption that each row has unique claim_id
    monthCounts = cndsdOffVisDf["CLM_FROM_DT_MONTH"].value_counts()                                             # assumption that each row has unique claim_id
    htmlList.append(plotly_plot(yearCounts,"hist","CLM_FROM_DT_YEAR",title="Claims Per Year"))
    htmlList.append(plotly_plot(monthCounts,"hist","CLM_FROM_DT_MONTH",title="Claims Per Month"))

    #### COSTS
    ### Overall distribution of claim costs
    htmlList.append(plotly_plot(cndsdOffVisDf,"hist","CLM_PMT_AMT"))
    ### Outlier search of claim costs
    htmlList.append(plotly_plot(cndsdOffVisDf,"violin","CLM_PMT_AMT"))
    ### Claim costs over months
    clmPerMonth = cndsdOffVisDf.groupby(["CLM_FROM_DT_MONTH"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmPerMonth,"bar","CLM_PMT_AMT"))
    ### Claim costs over years
    clmPerYear = cndsdOffVisDf.groupby(["CLM_FROM_DT_YEAR"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmPerYear,"bar","CLM_PMT_AMT"))
    
    
    costPMPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').DESYNPUF_ID.nunique()
    htmlList.append(plotly_plot(costPMPM,"bar",0,title="Claim Costs Per Member Per Month",labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Member Claim Cost($)'}))
    
    costPPPM=cndsdOffVisDf.groupby(['CLM_FROM_DT_MONTH'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_MONTH').PRVDR_NUM.nunique()
    htmlList.append(plotly_plot(costPPPM,"bar",0,title="Claim Costs Per Provider Per Month"),labels={'CLM_FROM_DT_MONTH':'Month','0':'Average Provider Claim Cost($)'})
    
    costPMPY=cndsdOffVisDf.groupby(['CLM_FROM_DT_YEAR'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_YEAR').DESYNPUF_ID.nunique()
    htmlList.append(plotly_plot(costPMPM,"bar",0,title="Claim Costs Per Member Per Year",labels={'CLM_FROM_DT_YEAR':'Year','0':'Average Member Claim Cost($)'}))
    
    costPPPY=cndsdOffVisDf.groupby(['CLM_FROM_DT_YEAR'])['CLM_PMT_AMT'].agg('sum')*1.0/cndsdOffVisDf.groupby('CLM_FROM_DT_YEAR').PRVDR_NUM.nunique()
    htmlList.append(plotly_plot(costPPPY,"bar",0,title="Claim Costs Per Provider Per Year",labels={'CLM_FROM_DT_YEAR':'Year','0':'Average Provider Claim Cost($)'}))
 
    print(f"{cndsdOffVisDf.DESYNPUF_ID.nunique()}")
    print(f"{cndsdOffVisDf.PRVDR_NUM.nunique()}")
    """
    """    
    ### Claim costs for each member
    clmCostByMember = offVisDf.groupby(["DESYNPUF_ID"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmCostByMember,"bar",clmCostByMember.index,"CLM_PMT_AMT"))
    ### Claim costs for each provider
    clmCostByProvider = cndsdOffVisDf.groupby(["PRVDR_NUM"])["CLM_PMT_AMT"].agg('sum')
    htmlList.append(plotly_plot(clmCostByProvider,"bar","CLM_PMT_AMT",title="Claim Costs Per Provider"))
    ### Per Claims Costs Per Beneficiary
    perClaimsCostPerBenef = offVisDf.groupby(['CLM_FROM_DT_MONTH'])['CLM_PMT_AMT'].agg('sum')*1.0/offVisDf.groupby('CLM_FROM_DT_MONTH').DESYNPUF_ID.nunique()
    print(perClaimsCostPerBenef)
    htmlList.append(plotly_plot(perClaimsCostPerBenef,"bar",clmCostByProvider.index,0))
    """
    ### Write out plotly visuzlizations to HTML file for interactive
    with open('/Users/cerebellum/Downloads/thfirst/fraud_visualizations.html', 'a') as f:
        for plotFig in htmlList:
            f.write(plotFig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()
    
    ### End analysis
    stdReport.flush()
    stdReport.close()

if __name__ == "__main__":
    main(sys.argv[1:])


    ### Inspect number of unique elements per column
    # unique_count = offVisDf.nunique(axis=0)
    ### Inspect unique elements per column (search for anomalous elements)
    # atd_dr = offVisDf["AT_PHYSN_NPI"].value_counts()
    # opr_dr = offVisDf["OP_PHYSN_NPI"].value_counts()
    # print(atd_dr)
    # print(opr_dr)
    #stdReport.write(unique_count)
    #stdReport.write(unique_table)