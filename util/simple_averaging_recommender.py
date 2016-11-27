import numpy as np
from recommender import Recommender
from collections import defaultdict

class SimpleAveragingRecommender(Recommender):
    def fit(self, X, y):
        self.tot_mean = np.mean(y)
        self.x0_means = defaultdict(int)
        self.x1_means = defaultdict(int)
        y_by_x0 = defaultdict(list)
        y_by_x1 = defaultdict(list)
        for i, y_value in enumerate(y):
            y_by_x0[X[i][0]].append(y_value)
            y_by_x1[X[i][1]].append(y_value)
        for x0, y_values in y_by_x0.items(): self.x0_means[x0] = np.mean(y_values) - self.tot_mean
        for x1, y_values in y_by_x1.items(): self.x1_means[x1] = np.mean(y_values) - self.tot_mean

    def predict(self, X):
        return np.array([self.x0_means[x0] + self.x1_means[x1] + self.tot_mean
            for x0, x1 in X])

