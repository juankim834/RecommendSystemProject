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
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astypr(int)

# Get main genre
movies['main_genre'] = movies['genres'].str.split('|').str[0]

# Filtering users who have less than 20 history and movies which have less than 5 history
counts_movies = movies.value_counts('movie_id')
movies = movies[counts_movies > 5].index.tolist()

counts_users = users.value_counts('user_id')
users = users[counts_users > 5]




