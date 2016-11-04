import pandas as pd
from collections import defaultdict

def time_augment_reviews(reviews):
    reviews['date']=pd.to_datetime(
            reviews.year*10000+reviews.month*100+reviews.day,
            format='%Y%m%d')

    ratings_by_user_and_date = defaultdict(lambda: defaultdict(list))
    for row in reviews[['user_id', 'date', 'stars']].itertuples():
        index, user_id, date, stars = row
        ratings_by_user_and_date[user_id][date].append(stars)

    reviews['n_given_same_day'] = [
        len(ratings_by_user_and_date[user_id][date])
        for user_id, date in reviews[['user_id','date']].values
    ]

    reviews['weekday'] = [date.weekday() for date in reviews['date']]
