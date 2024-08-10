import multiprocessing as mp
import pandas as pd
import os
import numpy as np
from itertools import product


def parallelize_df(df, func, n_cores=os.cpu_count(), **kwargs):
    '''
    Processes specified method on pandas dataframe using multiple cores

    Params:

        df (''pd.DataFrames''): Pandas dataframe to be processed
        func: Method to apply on provided dataframe
        n_cores (int): Number of processes that the mother process splits into

    Returns:

        ''pd.DataFrames''

    '''

    n_cores = min(n_cores, len(df))    
    other_args = [kwargs['{}'.format(m)] for m in func.__code__.co_varnames[1:]]
    df_split = np.array_split(df, n_cores)
    
    pool = mp.Pool(n_cores)
    #df = pd.concat(pool.map(func, product(df_split, *[[m] for m in other_args])))
    df_map = pool.starmap(func, product(df_split, *[[m] for m in other_args]))
    df = pd.concat(df_map)
    pool.close()
    pool.join()

    return df



def multicore(func, args, n_cores=os.cpu_count()):
    '''
    Processes specified method on a series of arguments in parallel. Number of cores is determined by
    whichever is smaller of the computer cores or the arguments nubmer.

    Params:

        fuc: function to apply on provided arguments
        args (list): a list of independent arguments

    '''

    n_cores = min(n_cores, len(args))
    pool = mp.Pool(processes=n_cores)
    pool.map(func, args)
    pool.close()
    pool.join()
