import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


movies = pd.read_csv(
    './data/ml-1m/movies.dat', 
    sep='::',
    header=None,
    names=['movie_id', 'title', 'genres'],
    engine='python',
    encoding='latin-1'
)

users = pd.read_csv(
    './data/ml-1m/users.dat',
    sep='::',
    header=None,
    names=['user_id', 'gender', 'age', 'occupation','zip'],
    engine='python'
)

ratings = pd.read_csv(
    './data/ml-1m/ratings.dat',
    sep='::',
    header=None,
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    engine='python'
)

# Release year from movie title
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(int)

# Filtering users who have less than 20 history and movies which have less than 5 history
movie_counts = ratings['movie_id'].value_counts()
valid_movie_ids = movie_counts[movie_counts >= 5].index
ratings = ratings[ratings['movie_id'].isin(valid_movie_ids)].copy()

user_counts = ratings['user_id'].value_counts()
valid_user_ids = user_counts[user_counts >= 20].index
ratings = ratings[ratings['user_id'].isin(valid_user_ids)].copy()

movies = movies[movies['movie_id'].isin(valid_movie_ids)].copy()
users = users[users['user_id'].isin(valid_user_ids)].copy()

# Get first three number of zipcode
users['zip_prefix'] = users['zip'].astype(str).str[:3]
# Encoding zipcode prefix
lbe_zip = LabelEncoder()
users['zip_enc'] = lbe_zip.fit_transform(users['zip_prefix']) + 1

lbe_occ = LabelEncoder()
users['occupation_enc'] = lbe_occ.fit_transform(users['occupation']) + 1

# Gender data processing
lbe_gender = LabelEncoder()
users['gender_enc'] = lbe_gender.fit_transform(users['gender'])
# 0:Female, 1:Male, or otherwise, depend on sequence of fit

# Age group processing
lbe_age = LabelEncoder()
users['age_enc'] = lbe_age.fit_transform(users['age']) + 1

# Merge data into ratings
ratings = ratings.merge(
    users[['user_id', 'gender_enc', 'age_enc', 'occupation_enc', 'zip_enc']], 
    on='user_id', 
    how='left'
)

# Label Encoding
# Mapping the incontinuous ids into continuous
lbe_user = LabelEncoder()
ratings['user_id_enc'] = lbe_user.fit_transform(ratings['user_id']) + 1 # Start from 1, 0 stand for unknown

lbe_movie = LabelEncoder()
ratings['movie_id_enc'] = lbe_movie.fit_transform(ratings['movie_id']) + 1

# Genre Processing
# Genre encoding
genre_set = set()
for x in movies['genres']:
    genre_set.update(x.split('|'))
genre2int = {val: ii+1 for ii, val in enumerate(genre_set)}

# Map MovieID to [Genre_ID_List]
# Transform genres in movie dataFrame into number list
movies['genre_ids'] = movies['genres'].apply(lambda x: [genre2int[g] for g in x.split('|')])

# Set up movie_id_enc -> genre_ids dictionary
movies['movie_id_enc'] = lbe_movie.transform(movies['movie_id']) + 1

# Generate a dict, example: {movie_enc_id: [1, 5, 2]}
movie_genre_dict = dict(zip(movies['movie_id_enc'], movies['genre_ids']))

# Pad all lists, make sure the length is 3
def get_padded_genres(genre_list):
    if len(genre_list) >= 3:
        return genre_list[:3]
    else:
        return genre_list + [0] * (3 - len(genre_list))
padded_movie_genre_dict = {k: get_padded_genres(v) for k, v in movie_genre_dict.items()}

ratings['genre_ids'] = ratings['movie_id_enc'].map(padded_movie_genre_dict)
movies['genre_ids'] = movies['movie_id_enc'].map(padded_movie_genre_dict)
movies.to_pickle("./data/cleaned/item_set.pkl")

ratings = ratings.merge(
    movies[['movie_id', 'release_year']], 
    on='movie_id',
    how='left'
)

# Label Generating
# Rating label 0 or 1, meaning a positive view of this rating activity. 
Threshold = 3
ratings['label'] = ratings['rating'].apply(lambda x: 1 if x >= Threshold else 0)
ratings = ratings.loc[ratings['label'] == 1]

# Time features processing
ratings['timestamp_dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['rating_hour'] = ratings['timestamp_dt'].dt.hour.astype(int) + 1
ratings['rating_weekday'] = ratings['timestamp_dt'].dt.weekday.astype(int) + 1
ratings['rating_month'] = ratings['timestamp_dt'].dt.month.astype(int)
ratings['rating_year'] = ratings['timestamp_dt'].dt.year + 1

BASE_YEAR = 1900
# Label Encoding for Year
def encode_year(year_series):
    encoded = year_series - BASE_YEAR + 1
    encoded = encoded.fillna(0)
    encoded[encoded < 0] = 0
    return encoded.astype(int)

ratings['year_enc'] = encode_year(ratings['rating_year'])
ratings['release_year_enc'] = encode_year(ratings['release_year'])

# # Calculate time lag features(Rating comment year - Movie release year)(Deprecated)
# ratings['time_lag'] = ratings['rating_year'] - ratings['release_year']
# ratings['time_lag'] = ratings['time_lag'].clip(lower=0)
# ratings['time_lag_norm'] = np.log1p(ratings['time_lag'])

ratings = ratings.sort_values(by=['user_id_enc', 'timestamp']).reset_index(drop=False)

from tqdm import tqdm
tqdm.pandas()

# Generate user's past 20 movies id list before this rating comment
def generate_history_movies(group):
    # Group: All the movement of a user
    movie_id_list = group['movie_id_enc'].tolist()
    hist_movie = []

    for i in range(len(movie_id_list)):
        if i == 0:
            hist_movie.append([0]*20)
        else:
            hist = movie_id_list[:i]
            if len(hist) > 20:
                hist = hist[-20:]
            else:
                hist = hist + [0] * (20 - len(hist))
            hist_movie.append(hist)
    
    group['hist_movie_ids'] = hist_movie
    return group

print("Generating history movies features...")

ratings = ratings.groupby('user_id_enc', as_index=False).progress_apply(generate_history_movies)

padded_movie_genre_dict[0] = [0, 0, 0]

# Generate user's past 20 movies' genres id list before this rating comment
def convert_hist_movies_to_genres(movie_id_list):
    genre_list = []
    for movie_id in movie_id_list:
        genres = padded_movie_genre_dict.get(movie_id, [0, 0, 0])
        genre_list.append(genres)
    return genre_list

print("Generating history genre features...")
ratings['hist_genre_ids'] = ratings['hist_movie_ids'].progress_apply(convert_hist_movies_to_genres)

# Spliting train test sets. 
# Sorting data
ratings['rank_latest'] = ratings.groupby(['user_id_enc'])['timestamp'].rank(method='first', ascending=False)

train = ratings[ratings['rank_latest'] > 2]
val = ratings[ratings['rank_latest'] == 2]
test = ratings[ratings['rank_latest'] == 1]

print("Calculate statistics features...")

# User Activity
train_user_activity = train['user_id_enc'].value_counts()
# Movie Popularity
train_movie_pop = train['movie_id_enc'].value_counts()
# Movie Avg Rating
train_movie_avg_rate = train.groupby('movie_id_enc')['rating'].mean()

# Global statistics(Use to cold startup or unknown data)
global_avg_rating = train['rating'].mean()

def add_stat_features(df):
    df_out = df.copy()

    # Mapping User Activity
    df_out['user_activity'] = df_out['user_id_enc'].map(train_user_activity).fillna(0)
    
    # Mapping Movie Popularity
    df_out['movie_pop'] = df_out['movie_id_enc'].map(train_movie_pop).fillna(0)

    # Mapping Movie Avg Rating
    df_out['movie_avg_rate'] = df_out['movie_id_enc'].map(train_movie_avg_rate).fillna(0)

    # Log normalization
    df_out['user_activity_log'] = np.log1p(df_out['user_activity'])
    df_out['movie_pop_log'] = np.log1p(df_out['movie_pop'])
    df_out['movie_avg_rate_log'] = np.log1p(df_out['movie_avg_rate'])

    return df_out

print("Applying features to Train/Val/Test...")
train = add_stat_features(train)
val = add_stat_features(val)
test = add_stat_features(test)

print("Check Train samples:")
print(train[['user_id_enc', 'user_activity_log', 'movie_pop_log', 'movie_avg_rate_log']].head())
print(train[['user_id_enc', 'hist_movie_ids', 'hist_genre_ids']].head())

print("\nCheck Test samples (Verification):")
print(test[['user_activity_log', 'movie_pop_log']].isna().sum())


train.to_pickle("./data/cleaned/train_set.pkl")
val.to_pickle("./data/cleaned/val_set.pkl")
test.to_pickle("./data/cleaned/test_set.pkl")

import pickle
with open("./data/cleaned/encoders.pkl", "wb") as f:
    pickle.dump({
        'user_encoder': lbe_user, 
        'movie_encoder': lbe_movie,
        'zip_encoder': lbe_zip,
        'gender_encoder': lbe_gender,
        'age_encoder': lbe_age,
        'occupation_encoder': lbe_occ,
        'genre_map': padded_movie_genre_dict,
        'genre_vocab_size': len(genre2int) + 1
    }, f)

