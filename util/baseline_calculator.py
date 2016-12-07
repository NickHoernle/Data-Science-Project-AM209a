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
            reviews.stars, self.baselines(reviews)))

    def baseline_stars(self, r):
        return self.global_mean + self.user_baselines[r.user_id] + self.busi_baselines[r.business_id]

    def baselines(self, reviews):
        return np.array([self.baseline_stars(r) for r in reviews.itertuples()])

    def transform(self, reviews, key='starz'):
        reviews[key] = [r.stars - self.baseline_stars(r) for r in reviews.itertuples()]
        return reviews

    def inverse_transform(self, reviews, predictions):
        return np.array([max(1, min(5, prediction + self.baseline_stars(r)))
            for prediction, r in zip(predictions, reviews.itertuples())])

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


def gradient_descent_minimize(loss_and_grad_fn, initial_guess, tol=50, maxiters=1000, learning_rate=0.00001, verbose=True):
    iters = 0
    state = initial_guess
    prev_loss = float('inf')
    prev_prev_loss = float('inf')

    while True:
        loss, grad = loss_and_grad_fn(state)
        state -= learning_rate * grad
        if verbose and iters % 25 == 0: print('loss at iter {}: {}'.format(iters, loss))
        if iters > maxiters: break
        if abs(loss - prev_loss) < tol: break
        if prev_loss < loss:
            learning_rate *= 0.9
            state += learning_rate * grad
            iters += 1
            continue
        elif loss/prev_loss > 0.9 and (iters < 25 or (loss-prev_loss > prev_loss-prev_prev_loss)):
            learning_rate *= 1.1
        prev_prev_loss = prev_loss
        prev_loss = loss
        iters += 1

    return state, loss, iters

class GradientDescentLinearRegressor():
    def predict(self, reviews):
        reviews['_intercept_mult'] = 1
        return np.dot(reviews[self.X_columns].values, self.coefs)

    def rmse(self, reviews):
        return np.sqrt(sklearn.metrics.mean_squared_error(reviews[self.y_column].values, self.predict(reviews)))

    def fit(self, reviews, X_columns, y_column, l2_penalty=10, l1_penalty=10, **kwargs):
        reviews['_intercept_mult'] = 1
        X_columns = ['_intercept_mult'] + list(X_columns)
        X = reviews[X_columns].values
        y = reviews[y_column].values
        self.X_columns = X_columns
        self.y_column = y_column

        def loss_and_grad(coefs):
            error = y - np.dot(X, coefs)
            nonzero = np.multiply(coefs, np.abs(coefs)>1e-9) # hack for making L1 work with floating point
            loss = np.sum(error**2) + l2_penalty*np.sum(coefs**2) + l1_penalty*np.sum(nonzero)
            grad = 2*l2_penalty*coefs + l1_penalty*np.sign(nonzero) - 2*np.dot(error, X)/float(X.shape[0])
            return loss, grad

        self.coefs, _loss, _iters = gradient_descent_minimize(loss_and_grad, np.zeros(len(X_columns)), **kwargs)

# eqn (4) from Netflix paper
class L2RegLeastSquaresBaselineCalculator(BaselineCalculator):
    def fit(self, reviews, l2_penalty=10, **kwargs):
        mu = reviews.stars.values.mean()
        self.global_mean = mu

        # we need to optimize with respect to a 1-D vector of parameters
        # containing [ user baselines ... business baselines ].
        # to compute the gradient with respect to these parameters, we need
        # to do a lot of mapping back and forth between ids and positions.
        user_ids = reviews.user_id.unique()
        busi_ids = reviews.business_id.unique()
        n_users = len(user_ids)
        n_busis = len(busi_ids)
        busi_ids_to_positions = {}
        user_ids_to_positions = {}
        for i, u in enumerate(user_ids): user_ids_to_positions[u] = i
        for i, b in enumerate(busi_ids): busi_ids_to_positions[b] = i+n_users
        user_positions = np.array([user_ids_to_positions[u] for u in reviews.user_id.values])
        busi_positions = np.array([busi_ids_to_positions[b] for b in reviews.business_id.values])

        # sparse representation
        coef_positions = [[] for _ in range(n_users + n_busis)]
        for i, (u, b) in enumerate(reviews[['user_id', 'business_id']].values):
            coef_positions[user_ids_to_positions[u]].append(i)
            coef_positions[busi_ids_to_positions[b]].append(i)
        for i, ps in enumerate(coef_positions):
            coef_positions[i] = np.array(ps)

        def loss_and_grad(coefs):
            error = (reviews.stars.values - mu
                     - coefs[user_positions]
                     - coefs[busi_positions])
            loss = (error**2).sum() + l2_penalty*(coefs**2).sum()
            grad = 2*(l2_penalty*coefs - np.array([error[cps].sum() for cps in coef_positions]))
            return loss, grad

        coefs, _loss, _iters = gradient_descent_minimize(loss_and_grad, np.zeros(n_users + n_busis), **kwargs)

        for i, u in enumerate(user_ids): self.user_baselines[u] = coefs[i]
        for i, b in enumerate(busi_ids): self.busi_baselines[b] = coefs[i+n_users]

class BusinessTimeBaselineCalculator(BaselineCalculator): # it is decidedly not business time
    def augment(self, reviews):
        raise NotImplementedError("implement me to set biztime")

    def baseline_stars(self, r):
        return self.global_mean + self.user_baselines[r.user_id] + self.busi_baselines[r.business_id] + self.biztime_baselines[r.biztime]

    def baselines(self, reviews):
        self.augment(reviews)
        return np.array([self.baseline_stars(r) for r in reviews.itertuples()])

    def fit(self, reviews, l2_penalty=10, **kwargs):
        mu = reviews.stars.values.mean()
        self.global_mean = mu
        self.biztime_baselines = defaultdict(int)
        self.augment(reviews)

        user_ids = reviews.user_id.unique()
        busi_ids = reviews.business_id.unique()
        biztimes = reviews.biztime.unique()
        n_users = len(user_ids)
        n_busis = len(busi_ids)
        n_biztimes = len(biztimes)
        busi_ids_to_positions = {}
        user_ids_to_positions = {}
        biztimes_to_positions = {}
        for i, u in enumerate(user_ids): user_ids_to_positions[u] = i
        for i, b in enumerate(busi_ids): busi_ids_to_positions[b] = i+n_users
        for i, t in enumerate(biztimes): biztimes_to_positions[t] = i+n_users+n_busis
        user_positions = np.array([user_ids_to_positions[u] for u in reviews.user_id.values])
        busi_positions = np.array([busi_ids_to_positions[b] for b in reviews.business_id.values])
        biztime_positions = np.array([biztimes_to_positions[t] for t in reviews.biztime.values])

        # sparse representation
        coef_positions = [[] for _ in range(n_users + n_busis + n_biztimes)]
        for i, (u, b, y) in enumerate(reviews[['user_id', 'business_id', 'biztime']].values):
            coef_positions[user_ids_to_positions[u]].append(i)
            coef_positions[busi_ids_to_positions[b]].append(i)
            coef_positions[biztimes_to_positions[y]].append(i)
        for i, ps in enumerate(coef_positions):
            coef_positions[i] = np.array(ps)

        def loss_and_grad(coefs):
            error = (reviews.stars.values - mu
                     - coefs[user_positions]
                     - coefs[busi_positions]
                     - coefs[biztime_positions])
            loss = (error**2).sum() + l2_penalty*(coefs**2).sum()
            grad = 2*(l2_penalty*coefs - np.array([error[cps].sum() for cps in coef_positions]))
            return loss, grad

        coefs, _loss, _iters = gradient_descent_minimize(loss_and_grad, np.zeros(len(coef_positions)), **kwargs)

        for i, u in enumerate(user_ids): self.user_baselines[u] = coefs[i]
        for i, b in enumerate(busi_ids): self.busi_baselines[b] = coefs[i+n_users]
        for i, t in enumerate(biztimes): self.biztime_baselines[t] = coefs[i+n_users+n_busis]

class BusinessYearBaselineCalculator(BusinessTimeBaselineCalculator):
    def augment(self, reviews):
        reviews['biztime'] = [round(float('{}.{}'.format(b, y)), 4) for b, y in reviews[['business_id', 'year']].values]

class Business6MonthBaselineCalculator(BusinessTimeBaselineCalculator):
    def augment(self, reviews):
        reviews['biztime'] = [round(float('{}.{}{}'.format(b, y, int(m<=6))), 5) for b, y, m in reviews[['business_id', 'year', 'month']].values]
