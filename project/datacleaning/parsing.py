import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import pad_sequences


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

# Get main genre
movies['main_genre'] = movies['genres'].str.split('|').str[0]

# Filtering users who have less than 20 history and movies which have less than 5 history
movie_counts = ratings['movie_id'].value_counts()
valid_movie_ids = movie_counts[movie_counts >= 5].index
ratings = ratings[ratings['movie_id'].isin(valid_movie_ids)].copy()

user_counts = ratings['user_id'].value_counts()
valid_user_ids = user_counts[user_counts >= 20].index
ratings = ratings[ratings['user_id'].isin(valid_user_ids)].copy()

movies = movies[movies['movie_id'].isin(valid_movie_ids)].copy()
users = users[users['user_id'].isin(valid_user_ids)].copy()



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

# Time features processing
ratings['timestamp_dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['hour'] = ratings['timestamp_dt'].dt.hour / 24.0
ratings['weekday'] = ratings['timestamp_dt'].dt.weekday / 6.0
ratings = ratings.sort_values(by=['user_id_enc', 'timestamp']).reset_index(drop=False)

from tqdm import tqdm
tqdm.pandas()

def generate_history(group):
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
                hist = [0] * (20 - len(hist)) + hist
            hist_movie.append(hist)
    
    group['hist_movie_ids'] = hist_movie
    return group

print("Generating sequence features...")

ratings = ratings.groupby('user_id_enc', as_index=False).progress_apply(generate_history)

# Merge all the features into Ratings list
ratings['target_genre_ids'] = ratings['movie_id_enc'].map(padded_movie_genre_dict)
# Label Generating
# Rating label 0 or 1, meaning a positive view of this rating activity. 
Threshold = 4
ratings['label'] = ratings['rating'].apply(lambda x: 1 if x >= Threshold else 0)

# Spliting train test sets. 
# Sorting data
ratings['rank_latest'] = ratings.groupby(['user_id_enc'])['timestamp'].rank(method='first', ascending=False)

train = ratings[ratings['rank_latest'] > 2]
val = ratings[ratings['rank_latest'] == 2]
test = ratings[ratings['rank_latest'] == 1]

train.to_pickle("./data/cleaned/train_set.pkl")
val.to_pickle("./data/cleaned/val_set.pkl")
test.to_pickle("./data/cleaned/test_set.pkl")

import pickle
with open("./data/cleaned/encoders.pkl", "wb") as f:
    pickle.dump({
        'user_encoder': lbe_user, 
        'movie_encoder': lbe_movie,
        'genre_map': padded_movie_genre_dict,
        'genre_vocab_size': len(genre2int) + 1
    }, f)
