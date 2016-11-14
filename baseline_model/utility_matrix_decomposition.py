import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from os import system
from os.path import isfile

# Using Vowpal Wabbit, following this example:
# https://github.com/JohnLangford/vowpal_wabbit/wiki/Matrix-factorization-example
#
# To run, you may need to first install Vowpal Wabbit. If you're on a mac and
# have Homebrew, you can just run
#
#  brew install vowpal-wabbit

reviews = pd.read_csv('yelp_dataset_challenge_academic_dataset/reviews.csv')
X = reviews[['user_id', 'business_id']].values
y = reviews['stars'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

def save_vw_matrix(filename, X, y):
    with open(filename, 'w') as f:
        for i in range(len(y)):
            f.write('{} |u {} |i {}\n'.format(y[i], *X[i]))

class VowpalWabbitRecommender():
    def __init__(self, model_name):
        self.model_name = model_name

    def fit(self, X, y, force=False):
        if not force and isfile('{}.vwmodel'.format(self.model_name)):
            print('Model already exists, skipping vowpal-wabbit fitting...')
            return
        assert(X.shape[1] == 2)
        save_vw_matrix('utility_train.vwmat', X, y)
        system('vw utility_train.vwmat -b 18 -q ui --rank 10 --l2 0.001 --learning_rate 0.015 --passes 20 --decay_learning_rate 0.97 --power_t 0 -f {}.vwmodel --cache_file {}.vwcache --quiet'.format(self.model_name, self.model_name))

    def predict(self, X):
        assert(X.shape[1] == 2)
        save_vw_matrix('utility_test.vwmat', X, np.ones(X.shape[0]))
        system('vw utility_test.vwmat -i {}.vwmodel -t --predictions predictions.vwout --quiet'.format(self.model_name))
        return np.genfromtxt('predictions.vwout')

    def score(self, X, y):
        preds = self.predict(X)
        y_avg = np.mean(y)
        tss = ((y - y_avg)**2).sum()
        sse = ((y - preds)**2).sum()
        return 1 - sse/tss

recommender = VowpalWabbitRecommender('factorization')
recommender.fit(X_train, y_train)
print recommender.score(X_test, y_test)
