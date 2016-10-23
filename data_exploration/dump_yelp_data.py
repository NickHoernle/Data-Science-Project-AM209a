import json
import os
import numpy
import pickle

yelp_data_dir = 'yelp_dataset_challenge_academic_dataset'

if not os.path.isdir(yelp_data_dir):
    raise ValueError('must download yelp data and move it to ./{}'.format(yelp_data_dir))

def each_yelp_entry_for(data_type):
    filename = 'yelp_academic_dataset_{}.json'.format(data_type)
    with open(os.path.join(yelp_data_dir, filename)) as f:
        for line in f:
            yield json.loads(line)

print('loading businesses')
biz_ids_to_indexes = {}
businesses = []
for i, b in enumerate(each_yelp_entry_for('business')):
    biz_ids_to_indexes[b['business_id']] = i
    b['business_id'] = i
    businesses.append(b)

print('loading users')
user_ids_to_indexes = {}
users = []
for i, u in enumerate(each_yelp_entry_for('user')):
    user_ids_to_indexes[u['user_id']] = i
    u['user_id'] = i
    users.append(u)
for user in users:
    user['friends'] = [user_ids_to_indexes[uid] for uid in user['friends']]

print('loading reviews')
reviews = []
for r in each_yelp_entry_for('review'):
    u_index = user_ids_to_indexes[r['user_id']]
    b_index = biz_ids_to_indexes[r['business_id']]
    n_stars = int(r['stars'])
    year, month, day = map(int, r['date'].split('-'))
    reviews.append([u_index, b_index, year, month, day, n_stars])
reviews = numpy.array(reviews)

print('dumping results')
pickle.dump(users,      open(os.path.join(yelp_data_dir, 'yelp_users.p'),      'wb'))
pickle.dump(reviews,    open(os.path.join(yelp_data_dir, 'yelp_reviews.p'),    'wb'))
pickle.dump(businesses, open(os.path.join(yelp_data_dir, 'yelp_businesses.p'), 'wb'))
