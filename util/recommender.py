import numpy as np
import sklearn.metrics as mets

class Recommender():
    def fit(self, X, y):
        raise NotImplementedError("must implement fit(self, X, y)")

    def predict(self, X):
        raise NotImplementedError("must implement predict(self, X)")

    def r2_score(self, X, y):
        return mets.r2_score(y, self.predict(X))

    def rmse(self, X, y):
        return np.sqrt(mets.mean_squared_error(y, self.predict(X)))

    def score(self, X, y):
        return self.rmse(X, y)
