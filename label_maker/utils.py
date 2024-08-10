import urllib.parse as urlparse
import pandas as pd
from smart_open import smart_open

def reads3csv_with_credential(old_url, aws_key, aws_secret):
    parsed = urlparse.urlparse(old_url)
    new_url = urlparse.urlunparse(parsed._replace(netloc="{}:{}@{}".format(aws_key, aws_secret, parsed.netloc)))
    df = pd.read_csv(smart_open(new_url))
    return df