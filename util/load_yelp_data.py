import os
import pandas
import numpy as np

# There is some python2 vs. python3 wonkiness to work around
try:
    import urllib
    download = urllib.urlretrieve
except AttributeError:
    import urllib.request
    download = urllib.request.urlretrieve

yelp_data_dir = '../yelp_dataset_challenge_academic_dataset'
s3_base_url = 'https://s3.amazonaws.com/ac209a-data'

# Classify the datapoints into the cities based on the latitude and the longitude
cities = np.array([['Edinburgh', 55.9533, -3.1883, 0, 0],
     ['Karlsruhe', 49.0069, 8.4037, 0, 0],
     ['Montreal', 45.5017, -73.5673, 0, 0],
     ['Waterloo', 43.4643, -80.5204, 0, 0],
     ['Pittsburgh', 40.4406, -79.9959, 0, 0],
     ['Charlotte', 35.2271, -80.8431, 0, 0],
     ['Urbana-Champaign', 40.1106, -88.2073, 0, -150000],
     ['Phoenix', 33.4484, -112.0740, 0, 0],
     ['Las Vegas', 36.1699, -115.1398, 0, 0],
     ['Madison', 43.0731, -89.4012, 0, 0]])

def location_of(pt):
    lat = np.array(cities[:,1], np.float32)
    lon = np.array(cities[:,2], np.float32)
    min_dist = np.argmin((pt['latitude'] - lat)**2 + (pt['longitude'] - lon)**2)
    return cities[min_dist, 0]

def load_yelp_dataframe(data_type):
    filename = '{}.csv'.format(data_type)
    fullpath = os.path.join(yelp_data_dir, filename)
    if not os.path.isfile(fullpath):
        print('{} not found locally, downloading from S3'.format(filename))
        os.system('mkdir -p {}'.format(yelp_data_dir))
        file_url = os.path.join(s3_base_url, filename)
        download(file_url, fullpath)
    df = pandas.read_csv(fullpath).drop('Unnamed: 0', 1)
    if data_type == 'businesses':
        df['location'] = df.apply(location_of, axis=1)
    return df

def filter_to_location(location, biz, rev, usr):
    lbiz = biz[biz.location == location]
    lrev = rev[rev.business_id.isin(lbiz.business_id.values)]
    lusr = usr[usr.user_id.isin(lrev.user_id.values)]
    return lbiz, lrev, lusr

def filter_to_categories(categories, biz, rev, usr):
    catfilter = lambda b: any(c in b.categories for c in categories)
    cbiz = biz[biz.apply(catfilter, axis=1)]
    crev = rev[rev.business_id.isin(cbiz.business_id.values)]
    cusr = usr[usr.user_id.isin(crev.user_id.values)]
    return cbiz, crev, cusr

def restaurants_and_bars_in(location, biz, rev, usr):
    return filter_to_categories(
            ['Restaurants', 'Bars', 'Nightlife'],
            *filter_to_location(location, biz, rev, usr))

def train_test_split_reviews(rev, seed=1000, train_size=0.7):
    n,m = rev.shape
    rev_trn = rev.sample(n=int(train_size*n), replace=False, random_state=seed)
    rev_tst = rev.drop(rev_trn.index)
    return rev_trn, rev_tst
