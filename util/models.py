import pandas as pd
import ast
import numpy as np

class simple_averaging:
    '''
    Most basic implementation of the yelp prediction of reiew score. The aim is to simply average 
    the typical scores that are given and make a prediction based on this average.
    '''
    def __init__(self, business_types=['restaurant'], location='Phoenix'):
        self.location = location
        self.business_types = [btype.lower() for btype in business_types]
        self.businesses = None
        self.model = 0
        
    def __intersect(self, a, b):
        """ return the intersection of two lists """
        a = [elem.lower() for elem in a]
        b = [elem.lower() for elem in b]
        return list(set(a) & set(b))

    def fit(self, X, y):
        '''
        Fit the model based on a training X and y
        '''
        city_df = X[X.location == self.location]
        city_df['chosen_category'] = city_df.apply(lambda x: len(self.__intersect(ast.literal_eval(x.categories), self.business_types)) > 0, axis=1)
        self.businesses = city_df[city_df.chosen_category == True]
        self.model = np.sum((y * self.businesses.review_count))/float(np.sum(self.businesses.review_count))
        return True

    def predict(self, X):
        '''
        Predict y, given an X matrix and assuming the model has been trained
        '''
        n,m = X.shape
        return np.array([self.model]*n)
    
    def score(self, X, y):
        '''
        Method to score model based on input X and given correct y
        '''
        SSE = np.sum((X.stars - self.model)**2)
        SST = np.sum((X.stars - np.mean(X.stars))**2)
        return 1-SSE/float(SST)