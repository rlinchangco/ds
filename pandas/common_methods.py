#!/usr/bin/python -tt

"""
To Add:
difference of dataframes based on column
addition of dataframes, horizontally and vertically
https://datagy.io/pandas-data-cleaning/
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


def drop_rows_by_string(df,col,string,exists=False):
    """
    drop rows from dataframe where string based column contains a specific string/substring
    """
    return df[df[col].str.contains(string) == exists]


def drop_rows_by_col_length(df,length,cols=[],sign='='):
    """
    drop rows from dataframe where string based column is greater than, equal to, or less than a certain length
    cols is a list of possible columns
    """
    # Figure out a way to let this be agnostic of column number https://newbedev.com/dynamically-filtering-a-pandas-dataframe
    for col in cols:
        df[col] = df[col].astype('str')
    mask = None
    if sign == '=':
        mask = (df[cols[0]].str.len() == length) & (df[cols[1]].str.len() == length)
    #print(df.shape)
    df = df.loc[mask]
    #print(df.shape)
    return df


def drop_rows_by_multicol_str_val(df,val,cols=[]):
    """
    drop rows from dataframe where string based column is greater than, equal to, or less than a certain length
    cols is a list of possible columns
    """
    # Figure out a way to let this be agnostic of column number https://newbedev.com/dynamically-filtering-a-pandas-dataframe
    for col in cols:
        df[col] = df[col].astype('str')
        # Try using .loc[row_indexer,col_indexer] = value instead
    mask = (df[cols[0]] != val) & (df[cols[1]] != val)
    df = df.loc[mask]
    return df


def find_unique_in_col(df,col=None):
    """
    find unique items in column(col=something) or all columns(col=None)
    """
    if col:
        print(df[col].unique())
    else:
        for column in df:
            print(df[column].unique())


def find_countunique_in_col(df,col=None):
    """
    find unique items in column(col=something) or all columns(col=None)
    """
    if col:
        print(df[col].nunique())
    else:
        for column in df:
            print(df[column].nunique())


def remove_duplicates(df1,column_name):
    """
    find and remove duplicate rows based on columns
    """
    cols = column_name.strip().split(',')
    newDF = df1.drop_duplicates(subset=cols)    
    return newDF


def df_join(df1,df2,column_name,other_column,joinType,joins,outputfile):
    """
    """
    df1[column_name] = df1[column_name].astype('str')
    df2[column_name] = df2[column_name].astype('str')
    if column_name and not other_column and joinType in joins:                                     # merge on column_name
        merged_df = df1.merge(df2,on=column_name,how=joinType)
    elif column_name and other_column and joinType in joins:                                     # merge on column_name
        merged_df = df1.merge(df2,left_on=column_name,right_on=other_column,how=joinType)
    else:                                                                           # merge on index
        merged_df = df1.join(df2,how=joinType)
    #print(merged_df)
    merged_df.to_excel(f"{outputfile}.{joinType}.xlsx",index=False)    


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
    column_name = None
    other_column = None
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
    # print(df1.shape)
    # length_filtered_df = drop_rows_by_col_length(df1,1,cols=['REF','ALT'])
    # print(length_filtered_df.shape)
    # val_filtered_df = drop_rows_by_multicol_str_val(length_filtered_df,string_row_drop,cols=['REF','ALT'])
    # print(val_filtered_df.shape)
    # val_filtered_df.to_csv(outputfile,index=False)
    # find_unique_in_col(df1,column_name)
    # find_countunique_in_col(df1,column_name)    
    
    cols = df1.columns.to_list()
    finalMeansDF = pd.DataFrame()
    newDF = pd.DataFrame()
    df2 = None
    if average != 0:
        cols = df1.columns.to_list()
        finalMeansDF = find_means(cols[1],cols[0],df1)
    if matchFile != '':
        df2 = multi_parse(matchFile)
    joins = ['left', 'right', 'outer', 'inner', 'cross']                            # potential join types
    if matchFile != '':
        df_join(df1,df2,column_name,other_column,joinType,joins,outputfile)
    if remove_dups:
        # find remove duplicate rows based on columns
        cols = column_name.strip().split(',')
        newDF = df1.drop_duplicates(subset=cols)
        print(newDF.shape)
    if transpose:
        thisdf = transpose_file(df1.set_index(column_name))
        thisdf.to_csv(f"{outputfile}.transposed.csv")
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
