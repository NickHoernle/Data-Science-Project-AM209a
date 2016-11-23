import numpy as np

class Recommender():
    def fit(self, X, y):
        raise NotImplementedError("must implement fit(self, X, y)")

    def predict(self, X):
        raise NotImplementedError("must implement predict(self, X)")

    def r2_score(self, X, y):
        preds = self.predict(X)
        y_avg = np.mean(y)
        tss = ((y - y_avg)**2).sum()
        sse = ((y - preds)**2).sum()
        return 1 - sse/tss

    def score(self, X, y):
        return self.r2_score(X, y)
