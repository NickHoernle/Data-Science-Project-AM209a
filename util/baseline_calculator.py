#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
import sklearn.metrics

class BaselineCalculator():
    def __init__(self):
        self.user_baselines = defaultdict(int)
        self.busi_baselines = defaultdict(int)

    def fit(self, reviews):
        raise NotImplementedError("implement me")

    def baseline_rmse(self, reviews): # assumes you've already fit
        return np.sqrt(sklearn.metrics.mean_squared_error(
            self.transform(reviews).starz, np.zeros(len(reviews))))

    def transform(self, reviews, key='starz'):
        reviews[key] = [
            max(1, min(5,
                s - self.global_mean - self.user_baselines[u] - self.busi_baselines[b]))
            for s, u, b in reviews[['stars', 'user_id', 'business_id']].values]
        return reviews

class AbsoluteMeanBaselineCalculator(BaselineCalculator):
    def fit(self, reviews):
        self.global_mean = np.mean(reviews.stars.values)

class SimpleAverageBaselineCalculator(BaselineCalculator):
    def fit(self, reviews):
        y = reviews.stars.values
        u = reviews.user_id.values
        b = reviews.business_id.values
        mu = np.mean(y)
        self.global_mean = mu
        yu = defaultdict(list)
        yb = defaultdict(list)
        for i, star in enumerate(y):
            yu[u[i]].append(star)
            yb[b[i]].append(star)
        for user, stars in yu.items():
            self.user_baselines[user] = np.mean(stars) - mu
        for busi, stars in yb.items():
            self.busi_baselines[busi] = np.mean(stars) - mu

# The idea here is to map ✮ => 0 and have it linearly increase to ✮✮✮✮✮ => 1.
# That converts our 5-star scale into a 0-1 scale.
# We then initially have a belief about a restaurant's rating centered at a value corresponding to the global mean,
# which we update by considering ✮✮✮✮✮ as V votes in favor, ✮ as V votes against,
# ✮✮✮ as V/2 votes in favor, V/2 votes against, etc.
class BetaPriorBaselineCalculator(BaselineCalculator):
    def fit(self, reviews, prior_strength=10, posterior_strength=1):
        def beta_mode(a, b):
            return (a - 1.0) / (a + b - 2.0)

        def beta_stars(a, b): # map from 0-1 scale to stars
            return beta_mode(a, b)*4 + 1

        y = reviews.stars.values
        u = reviews.user_id.values
        b = reviews.business_id.values
        mu = np.mean(y)
        self.global_mean = mu

        # If we want Beta(a,b) to have a mode corresponding to μ stars, and we know a+b=prior_strength,
        # the following computes the correct a and b
        beta_a = ((mu-1)/4.)*(prior_strength-2) + 1
        beta_b = prior_strength - beta_a
        if prior_strength > 2:
            assert(abs(beta_stars(beta_a, beta_b) - mu) < 0.0001)

        u_succs = defaultdict(float)
        u_fails = defaultdict(float)
        b_succs = defaultdict(float)
        b_fails = defaultdict(float)
        for i, star in enumerate(y):
            vote = posterior_strength*((star - 1.) / 4) # somewhat arbitrary
            u_succs[u[i]] += vote
            b_succs[b[i]] += vote
            u_fails[u[i]] += posterior_strength - vote
            b_fails[b[i]] += posterior_strength - vote
        for usr in u:
            self.user_baselines[usr] = beta_stars(beta_a + u_succs[usr], beta_b + u_fails[usr]) - mu
        for biz in b:
            self.busi_baselines[biz] = beta_stars(beta_a + b_succs[biz], beta_b + b_fails[biz]) - mu
