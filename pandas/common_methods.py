#!/usr/bin/python -tt

"""
To Add:
difference of dataframes based on column
addition of dataframes, horizontally and vertically
"""


import sys
import getopt
import os
import collections
import pandas as pd


def multi_parse(data_file, header_row=0):
    """
    Parses data file (accepts xlsx,tsv,csv) WITH header as first row.
    # read in chunks at a time(returns iterable if chunksize > 0)
    for df_chunk in pd.read_csv('data.csv', index_col=0, chunksize=8):
        print(df_chunk, end='\n\n')
    # find mean of duplicate IDs
    df.groupby('sample_id', as_index=False).mean()
    """
    df = pd.DataFrame()
    # reads csv, txt(tsv), and xlsx files
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file, header=0)
    elif data_file.endswith('.tsv') or data_file.endswith('.txt'):
        df = pd.read_csv(data_file, delimiter='\t', header=0)
    elif data_file.endswith('.xlsx'):
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


def find_means(groupColumn1,groupColumn2,df):
    """
    takes in pandas dataframe and finds means by grouping by columns (refactor)
    """
    meansDF = df.groupby(groupColumn1).mean()
    meansDF.index = meansDF.index.str.replace('_Pc_A','')
    meansDF.index = meansDF.index.str.replace('_Pc_B','')
    meansDF.index = meansDF.index.str.replace('_Pc_C','')
    finalMeansDF = meansDF.groupby(meansDF.index).mean()
    finalMeansDF.drop(columns=['rep','sub rep'],inplace=True)
    return finalMeansDF


def transpose_file(df):
    """
    simple transpose for column based files
    """
    return df.T


def drop_rows_by_string(df,col,string):
    """
    drop rows from dataframe where string based column contains a specific string/substring
    """
    return df[df[col].str.contains(string) == False]


def drop_rows_by_list(df,col,search_List,in_out='in'):
    """
    NEEDS MORE TESTING
    drop rows from dataframe where values are or are not in list (using ~)
    'in' returns df with rows matching list
    anything else returns df with rows not matching list
    """
    if in_out != 'in':
        return df[~df[col].isin(search_List)]
    else:
        return df[df[col].isin(search_List)]


def main(argv):
    """
    time python3 pandas_joins.py -i
    """
    inFile = ''
    outputfile = ''
    matchFile = ''
    column_name = ''
    joinType = 'left'
    average = 0
    transpose = None
    remove_dups = None
    string_row_drop = None
    list_drop = None
    search_list = None

    try:
        opts, args = getopt.getopt(
            argv, "hi:m:o:c:j:a:t:d:s:l:q:", ["inputFile=", "matchFile=", "ofile=","column=","joinType=","average=","transpose=","remove_dups=","string_row_drop=","list_drop=","search_list="])
    except getopt.GetoptError:
        print('pandas_joins.py -i <inputFile> -m <matchFile> -o <ofile> -c <column> -a <average>\n\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('pandas_joins.py -i <inputFile> -m <matchFile> -o <ofile> -c <column> -a <average>\n\n')
            sys.exit()
        elif opt in ("-i", "--inputFile"):
            inFile = arg
        elif opt in ("-m", "--matchFile"):
            matchFile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-j", "--joinType"):
            joinType = arg
        elif opt in ("-c", "--column"):
            column_name = arg        
        elif opt in ("-a", "--average"):
            average = arg
        elif opt in ("-t", "--transpose"):
            transpose = arg
        elif opt in ("-d", "--remove_dups"):
            remove_dups = arg
        elif opt in ("-s", "--string_row_drop"):
            string_row_drop = arg
        elif opt in ("-l", "--list_drop"):
            list_drop = arg
        elif opt in ("-q", "--search_list"):
            search_list = arg       # comma separated list     
            
    print('Input file is ', inFile)
    print('Output file is ', outputfile)
    df1 = multi_parse(inFile)
    ### Separate into specific functions
    cols = df1.columns.to_list()
    finalMeansDF = pd.DataFrame()
    newDF = pd.DataFrame()
    if average != 0:
        cols = df1.columns.to_list()
        finalMeansDF = find_means(cols[1],cols[0],df1)
    if matchFile != '':
        df2 = multi_parse(matchFile)
    joins = ['left', 'right', 'outer', 'inner', 'cross']                            # potential join types
    if matchFile != '':
        if column_name != '' and joinType in joins:                                     # merge on column_name
            merged_df = df1.merge(df2,on=column_name,how=joinType)
        else:                                                                           # merge on index
            merged_df = df1.join(df2,how=joinType)
        #print(merged_df)
        merged_df.to_excel(f"{outputfile}.{joinType}.xlsx",index=False)
    if remove_dups:
        # find remove duplicate rows based on columns
        cols = column_name.strip().split(',')
        newDF = df1.drop_duplicates(subset=cols)
        print(newDF.shape)
    if transpose:
        transpose_file(df1).to_excel(f"{outputfile}.transposed.xlsx")
    if string_row_drop:
        print(df1.shape)
        newDF = drop_rows_by_string(df1,column_name,string_row_drop)
        print(newDF.shape)
        newDF.to_excel(f"{outputfile}.removed.xlsx",index=False)
    if list_drop:
        print("HERE")
        search_list = search_list.strip().split(',')
        print(search_list)
        print(df1.shape)
        print(df1[column_name])
        newDF = drop_rows_by_list(df1,column_name,search_list,in_out='NOT')
        print(newDF.shape)
        newDF.to_excel(f"{outputfile}.removed.xlsx",index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
