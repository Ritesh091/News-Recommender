import numpy as np
import scipy
import pandas as pd
import math
import random

articles_df = pd.read_csv('shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
print(articles_df.head())

interactions_df = pd.read_csv('users_interactions.csv')
print(interactions_df.head())

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0,
   'BOOKMARK': 3.0,
   'FOLLOW': 5.0,
   'COMMENT CREATED': 4.5,
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
print(interactions_df)

users_interactions = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print(users_interactions)
print(len(users_interactions))
users_max_interactions = users_interactions[users_interactions >= 5].reset_index()[['personId']]
print(users_max_interactions)
print(len(users_max_interactions))

print(len(interactions_df))
interactions_users = interactions_df.merge(users_max_interactions, how = 'right',left_on = 'personId',right_on = 'personId')
print(interactions_users)
print(len(interactions_users))


def smooth_user_preference(x):
    return math.log(1 + x, 2)

interactions_full = interactions_users \
    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
    .apply(smooth_user_preference).reset_index()
print(len(interactions_full))
print(interactions_full)

item_popularity = interactions_full.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
print(item_popularity)

popular_articles = item_popularity.merge(articles_df)
print(popular_articles)
print(popular_articles['title'].head(20))