import boto3
import yaml
import click
import urllib.parse as urlparse

from label_maker import *

def run_site_selection(dir_config, run_local, do_corr, do_match, do_validate_metrics, do_selction):

    assert isinstance(run_local, bool)
    # params
    if dir_config.startswith("s3"):
        parsed = urlparse.urlparse(dir_config)
        config = yaml.load(boto3.resource('s3').Bucket(parsed.netloc).Object(parsed.path).get()['Body'].read())
    else:
        with open(dir_config, "r") as config:
            config = yaml.safe_load(config)
    params = {**config['Sites_Selection'], **config['AWS']}

    # calculate correlation
    if do_corr == True:
        get_corr(params, run_local=run_local)
    else:
        pass

    # match validation labels and predictions
    if do_match == True:
        match_validate_pred(params, run_local=run_local)
    else:
        pass

    # calculate validate metrics
    if do_validate_metrics:
        get_metrics(params, run_local=run_local)
    else:
        pass

    # select new sites
    if do_selction:
        get_new_sites(params, run_local=run_local)
    else:
        pass


@click.command()
@click.option('--dir-config', default='./config.yaml',
              help='Directory of config file')
@click.option('--run-local', is_flag=True,
              help='Whether the scripts are run in local conputer')
@click.option('--do-corr', is_flag=True,
              help='Whether to calculate the correlation')
@click.option('--do-match', is_flag=True,
              help='Whether to match directories of validate labels and score maps')
@click.option('--do-validate-metrics', is_flag=True,
              help='Whether to calculate metrics on prediction of each validation site')
@click.option('--do-selection', is_flag=True,
              help='Whether to select new sites based on calculated image correlation, prediciton metrics on validation sites,'
                   'and specified site numbers')
def main(dir_config, run_local, do_corr, do_match, do_validate_metrics, do_selection):
    run_site_selection(dir_config, run_local, do_corr, do_match, do_validate_metrics, do_selection)

if __name__=='__main__':
    main()



