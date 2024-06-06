'''
Handful of utility functions used throughout the repo
'''

import math
import pandas as pd

def merge_on_lat_lon(df1, df2, col_name=['cluster_lat', 'cluster_lon'], how='inner'):
    """
    Allows two dataframes to be merged on lat/lon
    Necessary because pandas has trouble merging on floats (understandably so)
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    # must use integers for merging, as floats induce errors
    df1['merge_lat'] = (10000 * df1[col_name[0]]).astype(int)
    df1['merge_lon'] = (10000 * df1[col_name[1]]).astype(int)
    
    df2['merge_lat'] = (10000 * df2[col_name[0]]).astype(int)
    df2['merge_lon'] = (10000 * df2[col_name[1]]).astype(int)
    
    df2.drop(col_name, axis=1, inplace=True)
    merged = pd.merge(df1, df2, on=['merge_lat', 'merge_lon'], how=how)
    merged.drop(['merge_lat', 'merge_lon'], axis=1, inplace=True)
    return merged

def create_space(lat, lon, d=10):
    """Creates a d km x d km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*d # roughly 0.045 for d=10
    return lat - v, lon - v, lat + v, lon + v
    