import os
import numpy as np
import pandas as pd
from rasterio.windows import Window
import itertools
import sys
sys.path.append("../")
from deeplearner import get_stacked_img, parallelize_df



def get_corr_df(catalog, buffer_a, buffer_b, chip_size, name1, name2):
    def getcoef_byrow(row):
        dirs_a = [row['dir_a_gs'], row['dir_a_os']]
        dirs_b = [row['dir_b_gs'], row['dir_b_os']]
        col_off_a = int(repr(row['col_a'])[-1]) * chip_size + buffer_a
        row_off_a = int(repr(row['row_a'])[-1]) * chip_size + buffer_a
        col_off_b = int(repr(row['col_b'])[-1]) * chip_size + buffer_b
        row_off_b = int(repr(row['row_b'])[-1]) * chip_size + buffer_b

        window_a = (col_off_a, row_off_a, chip_size, chip_size)
        window_b = (col_off_b, row_off_b, chip_size, chip_size)
        # get_stacked_img(imgPaths, usage, window=None)
        img_a = get_stacked_img(dirs_a, "train", window=window_a)
        img_b = get_stacked_img(dirs_b, "train", window=window_b)
        # img_a = get_img_stack(dirs_a)[:, row_off_a + buffer_a:row_off_a + buffer_a + chip_size,
        #         col_off_a + buffer_a:col_off_a + buffer_a + chip_size]
        # img_b = get_img_stack(dirs_b)[:, row_off_b + buffer_b: row_off_b + buffer_b + chip_size,
        #         col_off_b + buffer_b:col_off_b + buffer_b + chip_size]

        coef = np.corrcoef(img_a.reshape(-1), img_b.reshape(-1))[0, 1]
        return coef

    catalog = catalog.assign(coef=lambda df: df.apply(lambda row: getcoef_byrow(row), axis=1)) \
        .filter(items=[name1, name2, 'coef'])
    return catalog

def get_corr(params, run_local):

    # params
    dir_out = params['dir_correlation']
    fn_corr = params['fn_correlation']
    dir_data = params['dir_data']
    dir_csvs = params['dir_csvs']
    fn_catalog = params['catalog']
    fn_a = params['names_corr_a']
    fn_b = params['names_corr_b']
    name1, name2 = params['col_names_ab']
    buffer_a = params['buffer_a']
    buffer_b = params['buffer_b']
    grid_size = params['grid_size']
    dir_catalog = fn_catalog \
        if fn_catalog.startswith("s3://") or os.path.isabs(fn_catalog) \
        else os.path.join(dir_csvs, fn_catalog)
    dir_a = fn_a \
        if fn_a.startswith("s3://") or os.path.isabs(fn_a) \
        else os.path.join(dir_csvs, fn_a)
    dir_b = fn_b \
        if fn_b.startswith("s://") or os.path.isabs(fn_a) \
        else os.path.join(dir_csvs, fn_b)

    # table operation
    ## match image location for each grid in table_a and table_b
    catalog = pd.read_csv(dir_catalog)
    if run_local == True:
        catalog_a = pd.read_csv(dir_a) \
            .merge(catalog, how='inner', left_on=['col', 'row'], right_on=['col', 'row']) \
            .assign(url=lambda df: df['url']. \
                    map(lambda url: os.path.join(dir_data, "/".join(url.split("/")[-2:])))) \
            .filter(items=['col', 'row', 'name', 'season', 'url'])
        catalog_b = pd.read_csv(dir_b) \
            .merge(catalog, how='inner', left_on=['col', 'row'], right_on=['col', 'row']) \
            .assign(url=lambda df: df['url']. \
                    map(lambda url: os.path.join(dir_data, "/".join(url.split("/")[-2:])))) \
            .filter(items=['col', 'row', 'name', 'season', 'url'])

    else:
        catalog_a = pd.read_csv(dir_a) \
            .merge(catalog, how='inner', left_on=['col', 'row'], right_on=['col', 'row']) \
            .filter(items=['col', 'row', 'name', 'season', 'url'])
        catalog_b = pd.read_csv(dir_b) \
            .merge(catalog, how='inner', left_on=['col', 'row'], right_on=['col', 'row']) \
            .filter(items=['col', 'row', 'name', 'season', 'url'])

    catalog_a = catalog_a\
        .query("season == 'GS'")\
        .rename(columns={'url': 'dir_gs'})\
        .merge(catalog_a.query("season == 'OS'")\
            .rename(columns={'url': 'dir_os'}),
               left_on=['col', 'row', 'name'],
               right_on = ['col', 'row', 'name'])\
        .drop(columns=['season_x', 'season_y'])

    catalog_b = catalog_b\
        .query("season == 'GS'")\
        .rename(columns={'url': 'dir_gs'})\
        .merge(catalog_b.query("season == 'OS'")\
            .rename(columns={'url': 'dir_os'}),
               left_on = ['col', 'row', 'name'],
               right_on = ['col', 'row', 'name'])\
        .drop(columns=['season_x', 'season_y'])

    # pair table_a and table_b
    names_pair = list(itertools.product(catalog_a['name'], catalog_b['name']))
    catalog_paired = pd.DataFrame({
        name1: [m[0] for m in names_pair],
        name2: [m[1] for m in names_pair],
    }) \
        .merge(catalog_a, left_on=[name1], right_on=['name']) \
        .rename(columns={'col': 'col_a', 'row': 'row_a', 'dir_gs': 'dir_a_gs', 'dir_os': 'dir_a_os'}) \
        .merge(catalog_b, left_on=[name2], right_on=['name']) \
        .rename(columns={'col': 'col_b', 'row': 'row_b', 'dir_gs': 'dir_b_gs', 'dir_os': 'dir_b_os'}) \
        .drop(columns=['name_x', 'name_y'])

    # get correlation and write to file
    corr = parallelize_df(catalog_paired, get_corr_df,
                           buffer_a=buffer_a, buffer_b=buffer_b, chip_size=grid_size, name1=name1, name2=name2)
    corr.to_csv(os.path.join(dir_out, fn_corr), index=False)





