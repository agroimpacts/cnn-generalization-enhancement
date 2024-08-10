import os
import pandas as pd

from .utils import reads3csv_with_credential

def get_new_sites(params, run_local):

    dir_metrics = params['dir_metrics']
    fn_metrics = params['fn_metrics']
    dir_corr = params['dir_corr']
    fn_corr = params['fn_corr']
    dir_out = params['dir_new_sites']
    fn_out = params['fn_new_sites']

    num_sites = params['sites_number']
    val_sample_rate = params['validate_sample_rate']
    sample_metric = params['sample_criterion']
    metric_performance_relation = params['metric_performance_relation']

    assert metric_performance_relation in ['positive', 'negative']
    dir_metrics = fn_metrics if fn_metrics.startswith("s3") or os.path.isabs(fn_metrics) \
        else "{}/{}".format(dir_metrics, fn_metrics)
    dir_corr = fn_corr if fn_corr.startswith("s3") or os.path.isabs(fn_corr) \
        else "{}/{}".format(dir_corr, fn_corr)

    # read csvs
    if run_local:
        ACCESS_KEY_ID = params['aws_access']
        SECRET_ACCESS_KEY = params['aws_secret']
        metrics = reads3csv_with_credential(dir_metrics, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
            if dir_metrics.startswith("s3") else pd.read_csv(dir_metrics)
        corr = reads3csv_with_credential(dir_corr, ACCESS_KEY_ID, SECRET_ACCESS_KEY) \
            if dir_corr.startswith("s3") else pd.read_csv(dir_corr)
    else:
        metrics = pd.read_csv(dir_metrics)
        corr = pd.read_csv(dir_corr)

    # get new sites by specified validate samples and new sites
    val_sample_num = int(val_sample_rate * len(metrics))
    sample_per_val = int(num_sites / val_sample_num)
    if metric_performance_relation == "positive":
        new_sites = metrics \
            .nsmallest(val_sample_num, [sample_metric]) \
            .merge(corr, left_on=['name'], right_on=['name_val']) \
            .groupby('name_val') \
            .apply(lambda x: x.nlargest(sample_per_val, ['coef'])) \
            .reset_index(drop=True) \
            .filter(items=['name_pred'])
    else:
        new_sites = metrics \
            .nlargest(val_sample_num, [sample_metric]) \
            .merge(corr, left_on=['name'], right_on=['name_val']) \
            .groupby('name_val') \
            .apply(lambda x: x.nlargest(sample_per_val, ['coef'])) \
            .reset_index(drop=True) \
            .filter(items=['name_pred'])
    # write to csv
    new_sites.to_csv(os.path.join(dir_out, fn_out), index=False)
