import os
import numpy as np
from recommender import Recommender

def save_vowpal_wabbit_matrix(filename, X, y=None):
    if y is None: y = np.ones(X.shape[0])
    with open(filename, 'w') as f:
        for i in range(len(y)):
            f.write('{} |u {} |i {}\n'.format(y[i], *X[i]))

def vowpal_wabbit_installed():
    return os.system('which vw') == 0

class MatrixFactorizationRecommender(Recommender):
    def __init__(self, prefix='utility'):
        self.model_file = '{}.vwmodel'.format(prefix)
        self.cache_file = '{}.vwcache'.format(prefix)
        self.train_file = '{}_train.vwmat'.format(prefix)
        self.test_file = '{}_X_test.vwmat'.format(prefix)
        self.pred_file = '{}_y_test.vwout'.format(prefix)

    def fit(self, X, y, force=False):
        # if we already have a model file, don't necessarily retrain
        if not force and os.path.isfile(self.model_file):
            print('Model file already exists, skipping vowpal-wabbit fitting'); return
        # ensure vowpal-wabbit is installed
        if not vowpal_wabbit_installed(): 
            print('vw not installed; run brew install vowpal-wabbit (if on a mac)'); return
        # should just have users and businesses
        assert(X.shape[1] == 2)
        # save the utility matrix for X, y
        save_vowpal_wabbit_matrix(self.train_file, X, y)
        # fit the model
        os.system('vw {} -b 18 -q ui --rank 10 --l2 0.001 \
            --learning_rate 0.015 --passes 20 --decay_learning_rate 0.97 \
            --power_t 0 -f {} --cache_file {} --quiet'.format(
              self.train_file, self.model_file, self.cache_file))

    def predict(self, X):
        if not vowpal_wabbit_installed(): 
            print('vw not installed; run brew install vowpal-wabbit (if on a mac)'); return np.zeros(X.shape[0])
        # should just have users and businesses
        assert(X.shape[1] == 2)
        # save the utility matrix for X
        save_vowpal_wabbit_matrix(self.test_file, X)
        # dump predictions
        os.system('vw {} -i {} -t --predictions {} --quiet'.format(
          self.test_file, self.model_file, self.pred_file))
        # load them
        return np.genfromtxt(self.pred_file)
