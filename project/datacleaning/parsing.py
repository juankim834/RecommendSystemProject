import pandas as pd


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
ratings = ratings[ratings['movie_id'].isin(valid_movie_ids)]

user_counts = ratings['user_id'].value_counts()
valid_user_ids = user_counts[user_counts >= 20].index
ratings = ratings[ratings['user_id'].isin(valid_user_ids)]

movies = movies[movies['movie_id'].isin(valid_movie_ids)]
users = users[users['user_id'].isin(valid_user_ids)]

# Label Generating

Threshold = 4
ratings['label'] = ratings['rating'].apply(lambda x: 1 if x >= Threshold else 0)

# Spliting train test sets. 
# Sorting data

ratings = ratings.sort_values(by=['user_id', 'timestamp'])
# Make indicators
ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)

train = ratings[ratings['rank_latest'] > 2]
val = ratings[ratings['rank_latest'] == 2]
test = ratings[ratings['rank_latest'] == 1]

print(f"Train shape: {train.shape}")
print(f"Val shape: {val.shape}")
print(f"Test shape: {test.shape}")

ratings.to_csv("./data/df_ratings.csv")

train.to_csv("./data/cleaned/df_train.csv", index=False)
val.to_csv("./data/cleaned/df_val.csv", index=False)
test.to_csv("./data/cleaned/df_test.csv", index=False)
