import os
import sys
import numpy as np
import rasterio
import pandas as pd
import math as ma


sys.path.append("../")
from .utils import reads3csv_with_credential
from deeplearner import parallelize_df, load_data, BinaryMetrics


def match_validate_pred(params, run_local):
    dir_csvs = params['dir_csvs']
    dir_pred_catalogs = params['dir_pred_catalogs']
    fn_pred_catalog_format = params['fn_pred_catalog_format']
    dir_pred_format = params['dir_pred_format']
    dir_label_format = params['dir_label_format']
    aois = eval(params['aois'])
    names_validate = params['names_validate']
    dir_out = params['dir_val_pred_matched']
    fn_out = params['fn_val_pred_matched']
    # aws
    ACCESS_KEY_ID = params['aws_access']
    SECRET_ACCESS_KEY = params['aws_secret']
    REGION = params['aws_region']

    dir_names_validate = names_validate if names_validate.startswith("s3") or os.path.isabs(names_validate) \
        else "{}/{}".format(dir_csvs, names_validate)
    if run_local:

        names_validate = reads3csv_with_credential(dir_names_validate, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
            if dir_names_validate.startswith("s3") else pd.read_csv(dir_names_validate) \
            .assign(tile_col=lambda df: df['col'].map(lambda col: ma.floor(col / 10))) \
            .assign(tile_row=lambda df: df['row'].map(lambda row: ma.floor(row / 10)))

        for aoi in aois:
            fn_pred_catalog = fn_pred_catalog_format.format(aoi)
            dir_pred_catalog = "{}/{}".format(dir_pred_catalogs, fn_pred_catalog)
            # whether read from s3
            pred_catalog_slice = reads3csv_with_credential(dir_pred_catalog, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
                if dir_pred_catalog.startswith("s3") else pd.read_csv(dir_pred_catalog)
            pred_catalog_slice = pred_catalog_slice \
                .query("type == 'center'") \
                .assign(aoi=aoi) \
                .merge(names_validate, how="inner", left_on=['tile_col', 'tile_row'], right_on=['tile_col', 'tile_row'])
            pred_catalog = pred_catalog.append(pred_catalog_slice) \
                if "pred_catalog" in locals() else pred_catalog_slice

    else:

        names_validate = pd.read_csv(dir_names_validate) \
            if dir_names_validate.startswith("s3") else pd.read_csv(dir_names_validate) \
            .assign(tile_col=lambda df: df['col'].map(lambda col: ma.floor(col / 10))) \
            .assign(tile_row=lambda df: df['row'].map(lambda row: ma.floor(row / 10)))

        for aoi in aois:
            fn_pred_catalog = fn_pred_catalog_format.format(aoi)
            dir_pred_catalog = "{}/{}".format(dir_pred_catalogs, fn_pred_catalog)

            # read table
            pred_catalog_slice = pd.read_csv(dir_pred_catalog)
            pred_catalog_slice = pred_catalog_slice \
                .query("type == 'center'") \
                .assign(aoi=aoi) \
                .merge(names_validate, how="inner", left_on=['tile_col', 'tile_row'], right_on=['tile_col', 'tile_row'])
            pred_catalog = pred_catalog.append(pred_catalog_slice) \
                if "pred_catalog" in locals() else pred_catalog_slice

    pred_catalog = pred_catalog \
        .assign(dir_pred=lambda df: df.apply(
        lambda row: dir_pred_format.format(row['aoi'], row['tile_col'], row['tile_row']), axis=1)) \
        .assign(
        dir_label=lambda df: df.apply(lambda row: dir_label_format.format(row['name'], row['col'], row['row']),
                                      axis=1)) \
        .filter(items=['name', 'col', 'row', 'dir_pred', 'dir_label'])
    pred_catalog.to_csv("{}/{}".format(dir_out, fn_out), index=False)



def get_metrics_df(dataframe, chip_size):
    def get_metrics_byrow(row):
        dir_pred = row['dir_pred']
        dir_label = row['dir_label']
        col_off_pred = int(repr(row['col'])[-1]) * chip_size
        row_off_pred = int(repr(row['row'])[-1]) * chip_size

        pred = load_data(dir_pred, isLabel=True)[row_off_pred: row_off_pred + chip_size,
               col_off_pred: col_off_pred + chip_size]
        label = load_data(dir_label, isLabel=True)

        return BinaryMetrics(np.where(label == 1, 1, 0),
                             np.where(pred == 1, 1, 0))

    dataframe = dataframe \
        .assign(metrics=lambda df: df.apply(lambda row: get_metrics_byrow(row), axis=1)) \
        .assign(accuracy=lambda df: df.apply(lambda row: row['metrics'].accuracy(),
                                             axis=1)) \
        .assign(precision=lambda df: df.apply(lambda row: row['metrics'].precision(),
                                              axis=1)) \
        .assign(recall=lambda df: df.apply(lambda row: row['metrics'].recall(), axis=1)) \
        .assign(fpr=lambda df: df.apply(lambda row: row['metrics'].false_positive_rate(),
                                        axis=1)) \
        .filter(items=['name', 'col', 'row', 'accuracy', 'precision', 'recall', 'fpr'])
    return dataframe

def get_metrics(params, run_local):

    dir_out = params['dir_metrics']
    fn_out = params['fn_metrics']
    dir_matched = params['dir_val_pred_matched']
    fn_matched = params['fn_val_pred_matched']
    dir_pred_catalog = "{}/{}".format(dir_matched, fn_matched)

    # get metrics
    if run_local == True:
        pred_catalog = reads3csv_with_credential(dir_pred_catalog, params['aws_access'], params['aws_secret']) \
            if dir_pred_catalog.startswith("s3") else pd.read_csv(dir_pred_catalog)
        # aws
        ACCESS_KEY_ID = params['aws_access']
        SECRET_ACCESS_KEY = params['aws_secret']
        REGION = params['aws_region']
        Session = rasterio.env.Env(aws_acess_key_id = ACCESS_KEY_ID,
                                   aws_secret_access_key=SECRET_ACCESS_KEY,
                                   region_name=REGION)
        with Session:
            validate_metrics = parallelize_df(pred_catalog, get_metrics_df, chip_size=params['grid_size'])

    else:
        pred_catalog = pd.read_csv(dir_pred_catalog)
        validate_metrics = parallelize_df(pred_catalog, get_metrics_df, chip_size=params['grid_size'])

    validate_metrics.to_csv("{}/{}".format(dir_out, fn_out), index=False)
