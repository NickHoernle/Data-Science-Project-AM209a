import pickle
import os
import numpy

# There is some python2 vs. python3 wonkiness to work around
try:
    import urllib
    url_retrieve = urllib.urlretrieve
except AttributeError:
    import urllib.request
    url_retrieve = urllib.request.urlretrieve

yelp_data_dir = '../yelp_dataset_challenge_academic_dataset'
s3_base_url = 'https://s3.amazonaws.com/ac209a-data'

def load_yelp_pickle(data_type):
    filename = 'yelp_{}.p'.format(data_type)
    fullpath = os.path.join(yelp_data_dir, filename)

    if not os.path.isfile(fullpath):
        print('{} not found locally, downloading from S3'.format(filename))
        os.makedirs(yelp_data_dir, exist_ok=True)
        file_url = os.path.join(s3_base_url, filename)
        url_retrieve(file_url, fullpath)

    with open(fullpath, 'rb') as f:
        data = pickle.load(f)

    return data
