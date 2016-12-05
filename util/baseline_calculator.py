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
        stars = self.inverse_transform(reviews, np.zeros(len(reviews)))
        return np.sqrt(sklearn.metrics.mean_squared_error(
            reviews.stars, stars))

    def transform(self, reviews, key='starz'):
        reviews[key] = [
            s - self.global_mean - self.user_baselines[u] - self.busi_baselines[b]
            for s, u, b in reviews[['stars', 'user_id', 'business_id']].values]
        return reviews

    def inverse_transform(self, reviews, predictions):
        return np.array([
            max(1, min(5,
            p + self.global_mean + self.user_baselines[u] + self.busi_baselines[b]))
            for p, (u, b) in zip(predictions, reviews[['user_id', 'business_id']].values)
        ])

    def fit_transform(self, reviews, key='starz'):
        self.fit(reviews)
        return self.transform(reviews, key=key)

class AbsoluteMeanBaselineCalculator(BaselineCalculator):
    def fit(self, reviews):
        self.global_mean = np.mean(reviews.stars.values)

# eqn (1) from Netflix paper
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

# eqns (2) and (3) from Netflix paper
class DecoupledRegularizedBaselineCalculator(BaselineCalculator):
    def fit(self, reviews, busi_reg_strength=2.5, user_reg_strength=5):
        y = reviews.stars.values
        u = reviews.user_id.values
        b = reviews.business_id.values
        mu = np.mean(y)
        self.global_mean = mu
        yu = defaultdict(list)
        yb = defaultdict(list)
        for i, star in enumerate(y):
            yb[b[i]].append(star)
            yu[u[i]].append((star, b[i]))
        for busi, stars in yb.items():
            self.busi_baselines[busi] = (
                sum(star - mu for star in stars) /
                float(busi_reg_strength + len(stars)))
        for user, star_biz_pairs in yu.items():
            self.user_baselines[user] = (
                sum(star - mu - self.busi_baselines[biz] for star, biz in star_biz_pairs) /
                float(user_reg_strength + len(star_biz_pairs)))

# eqn (4) from Netflix paper
class L2RegLeastSquaresBaselineCalculator(BaselineCalculator):
    def fit(self, reviews, l2_pen=10, tol=1, maxiters=1000, learn_rate=0.00001, verbose=True):
        user_ids = reviews.user_id.unique()
        busi_ids = reviews.business_id.unique()
        self.global_mean = reviews.stars.values.mean()
        normed_stars = reviews.stars.values - self.global_mean

        busi_indexes = {}
        user_indexes = {}
        n_users = len(user_ids)
        n_busis = len(busi_ids)
        for i, u in enumerate(user_ids): user_indexes[u] = i
        for i, b in enumerate(busi_ids): busi_indexes[b] = i
        uids = np.array([user_indexes[u] for u in reviews.user_id.values])
        bids = np.array([busi_indexes[b] for b in reviews.business_id.values])
        busi_indexes = None
        user_indexes = None
        coids = [np.argwhere(reviews.user_id.values == usr) for usr in user_ids] + \
                [np.argwhere(reviews.business_id.values == biz) for biz in busi_ids]

        iters = 0
        coefs = np.zeros(n_users + n_busis)
        prev_loss = float('inf')

        while True:
            error = (normed_stars - coefs[:n_users][uids] - coefs[n_users:][bids])
            loss = np.sum(error**2) + l2_pen*np.sum(coefs**2)
            grad = 2*l2_pen*coefs - 2*np.array([np.sum(error[coid]) for coid in coids])
            coefs -= learn_rate * grad
            if verbose and iters % 10 == 0: print(loss)
            if iters > maxiters: break
            if abs(loss - prev_loss) < tol: break
            prev_loss = loss
            iters += 1

        for i, u in enumerate(user_ids):
            self.user_baselines[u] = coefs[:n_users][i]
        for i, b in enumerate(busi_ids):
            self.busi_baselines[b] = coefs[n_users:][i]
