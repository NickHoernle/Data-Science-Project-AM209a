import os
import pandas

# There is some python2 vs. python3 wonkiness to work around
try:
    import urllib
    download = urllib.urlretrieve
except AttributeError:
    import urllib.request
    download = urllib.request.urlretrieve

yelp_data_dir = '../yelp_dataset_challenge_academic_dataset'
s3_base_url = 'https://s3.amazonaws.com/ac209a-data'

def load_yelp_dataframe(data_type):
    filename = '{}.csv'.format(data_type)
    fullpath = os.path.join(yelp_data_dir, filename)

    if not os.path.isfile(fullpath):
        print('{} not found locally, downloading from S3'.format(filename))
        os.system('mkdir -p {}'.format(yelp_data_dir))
        file_url = os.path.join(s3_base_url, filename)
        download(file_url, fullpath)

    return pandas.read_csv(filename)
