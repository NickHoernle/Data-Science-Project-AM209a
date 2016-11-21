import pandas as pd
import ast
import numpy as np

def filter(dataframe, filter_attributes = {'categories':['Restaurants', 'Fast Food']}, and_or_flag = 'AND'):
    '''
    Method to filter a given dataframe based on an input dictionary of column to attribute mapping.
    Input:
        dataframe: the dataframe that you want to filter
        filter_attributes: the dictionary of column to attributes that you want to filter on
        and_or_flag: for multiple columns are we filtering on an 'and' or an 'or' relationship
    Output:
        filtered_dataframe
    '''
    def __intersect(a, b):
        """ return the intersection of two lists """
        a = [elem.lower() for elem in a]
        b = [elem.lower() for elem in b]
        return list(set(a) & set(b))
    def __handle_filter(request):
        input_string = ''
        try:
            input_string = ast.literal_eval(str(request[category]))
        except:
            input_string = ast.literal_eval('"'+str(request[category])+'"')
        if not type(input_string) == list:
            input_string = [input_string]
        return len(__intersect(input_string, attributes)) > 0

    # check that the dataframe has the columns specified
    for category in filter_attributes:
        if category not in dataframe.columns:
            raise ValueError('Incorrect column names')
    if and_or_flag == 'AND':
        selections = pd.Series(index=dataframe.index, data=1, dtype=np.int8)
        for category in filter_attributes:
            attributes = filter_attributes[category]
            if not type(attributes) == list:
                attributes = [attributes]
            selection = pd.Series(index=dataframe.index, data=dataframe.apply( __handle_filter, axis=1 ), dtype=np.int8)
            selections = np.bitwise_and(selections, selection)
        return dataframe.iloc[selection[selections == 1].index]
            
    elif and_or_flag == 'OR':
        return 'Here'
    raise ValueError('Incorrect and/or flag supplied. Please use "AND" and "OR"')
    
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