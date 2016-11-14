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

def load_yelp_dataframe(data_type):
    filename = '{}.csv'.format(data_type)
    fullpath = os.path.join(yelp_data_dir, filename)

    if not os.path.isfile(fullpath):
        print('{} not found locally, downloading from S3'.format(filename))
        os.system('mkdir -p {}'.format(yelp_data_dir))
        file_url = os.path.join(s3_base_url, filename)
        download(file_url, fullpath)

    return pandas.read_csv(fullpath)

def create_standard_train_and_test(
            dataframes,
            location='Phoenix',
            train_size = 0.7,
            seed=1000,
        ):
    '''
    Accepts:
     - dataframe=[businesses,users,reviews], array of input dataframes,
     - location='Phoenix', the location that we want to filter on,
     - train_size=0.7, the relative size of the training set
     - seed=1000, the random seed to ensure we all have the same train/test sets
    '''
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
    
    test_train_tuples = []
    for df in dataframes:
        if 'type' in df.columns:
            if df.iloc[0]['type'] == 'business':
                lat = np.array(cities[:,1], np.float32)
                lon = np.array(cities[:,2], np.float32)
                x_offsets = np.array(cities[:,3], np.float32)
                y_offsets = np.array(cities[:,4], np.float32)

                df['location'] = df.apply(classify_datapoint_by_latitude_and_longitude, args=[lat, lon, cities], axis=1)
                
                df = df[df.location == location]
        
        n,m = df.shape
        train = df.sample(n=int(train_size*n), replace=False, random_state=seed)
        test = df.drop(train.index)
        test_train_tuples.append((train, test))
    return test_train_tuples


def classify_datapoint_by_latitude_and_longitude(datapoint, lat, lon, cities):
    #     print datapoint
    latitude = datapoint['latitude']
    longitude = datapoint['longitude']
    distances = (latitude - lat)**2 + (longitude - lon)**2
    min_dist = np.argmin(distances)
    return cities[min_dist, 0]
