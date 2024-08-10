import boto3
import yaml
import click
import urllib.parse as urlparse

from label_maker import *


def run_rasterization(dir_config, run_local):

    assert isinstance(run_local, bool)
    # params
    if dir_config.startswith("s3"):
        parsed = urlparse.urlparse(dir_config)
        config = yaml.load(boto3.resource('s3').Bucket(parsed.netloc).Object(parsed.path).get()['Body'].read())

    else:
        with open(dir_config, "r") as config:
            config = yaml.safe_load(config)
    # useful params
    params = {**config['Rasterize_Labels'], **config['AWS']}

    get_rasterization(params, run_local=run_local)


@click.command()
@click.option('--dir-config', default='./config.yaml',
              help='Directory of config file')
@click.option('--run-local', is_flag=True,
              help='Whether the scripts are run in local conputer')
def main(dir_config, run_local):
    run_rasterization(dir_config, run_local)

if __name__=='__main__':
    main()
