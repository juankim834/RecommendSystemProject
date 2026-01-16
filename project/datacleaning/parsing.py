import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from tqdm import tqdm

tqdm.pandas()

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================
print("Loading raw data...")
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
    names=['user_id', 'gender', 'age', 'occupation', 'zip'],
    engine='python'
)

ratings = pd.read_csv(
    './data/ml-1m/ratings.dat',
    sep='::',
    header=None,
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    engine='python'
)

print(f"Raw data - Users: {len(users)}, Movies: {len(movies)}, Ratings: {len(ratings)}")

# ============================================================================
# 2. FILTER DATA (min interactions)
# ============================================================================
print("\nFiltering sparse users/movies...")

# Keep movies with at least 5 ratings
movie_counts = ratings['movie_id'].value_counts()
valid_movie_ids = movie_counts[movie_counts >= 5].index
ratings = ratings[ratings['movie_id'].isin(valid_movie_ids)].copy()

# Keep users with at least 20 ratings
user_counts = ratings['user_id'].value_counts()
valid_user_ids = user_counts[user_counts >= 20].index
ratings = ratings[ratings['user_id'].isin(valid_user_ids)].copy()

# Filter movies and users tables
movies = movies[movies['movie_id'].isin(valid_movie_ids)].copy()
users = users[users['user_id'].isin(valid_user_ids)].copy()

print(f"After filtering - Users: {len(users)}, Movies: {len(movies)}, Ratings: {len(ratings)}")

# ============================================================================
# 3. PROCESS MOVIES
# ============================================================================
print("\nProcessing movie features...")

# Extract release year from title
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

# Process genres
genre_set = set()
for genres_str in movies['genres']:
    genre_set.update(genres_str.split('|'))

# Create genre vocabulary (1-indexed, 0 for padding)
genre2int = {genre: idx + 1 for idx, genre in enumerate(sorted(genre_set))}

# Convert genre strings to ID lists
movies['genre_ids_raw'] = movies['genres'].apply(
    lambda x: [genre2int[g] for g in x.split('|')]
)

# ============================================================================
# 4. ENCODE IDs (1-indexed, 0 reserved for padding/unknown)
# ============================================================================
print("\nEncoding categorical features...")

# Movie ID encoding
lbe_movie = LabelEncoder()
movies['movie_id_enc'] = lbe_movie.fit_transform(movies['movie_id']) + 1

# User features encoding
lbe_user = LabelEncoder()
users['user_id_enc'] = lbe_user.fit_transform(users['user_id']) + 1

lbe_gender = LabelEncoder()
users['gender_enc'] = lbe_gender.fit_transform(users['gender']) + 1

lbe_age = LabelEncoder()
users['age_enc'] = lbe_age.fit_transform(users['age']) + 1

lbe_occ = LabelEncoder()
users['occupation_enc'] = lbe_occ.fit_transform(users['occupation']) + 1

# Zip code prefix (first 3 digits)
users['zip_prefix'] = users['zip'].astype(str).str[:3]
lbe_zip = LabelEncoder()
users['zip_enc'] = lbe_zip.fit_transform(users['zip_prefix']) + 1

# ============================================================================
# 5. BUILD MOVIE GENRE MAPPING
# ============================================================================
print("\nBuilding movie-genre mappings...")

# Create padded genre list (fixed length 3)
def pad_genres(genre_list, max_len=3):
    if len(genre_list) >= max_len:
        return genre_list[:max_len]
    return genre_list + [0] * (max_len - len(genre_list))

movies['genre_ids'] = movies['genre_ids_raw'].apply(lambda x: pad_genres(x, 3))

# Create mapping: movie_id_enc -> padded genre list
movie_genre_dict_padded = dict(zip(movies['movie_id_enc'], movies['genre_ids']))
movie_genre_dict_padded[0] = [0, 0, 0]  # For padding movie ID

# Create mapping: movie_id_enc -> set of genre IDs (for hard negative matching)
movie_genre_dict_set = dict(zip(movies['movie_id_enc'], 
                                movies['genre_ids_raw'].apply(set)))
movie_genre_dict_set[0] = set()  # For padding movie ID

# ============================================================================
# 6. MERGE AND ENRICH RATINGS
# ============================================================================
print("\nMerging features into ratings...")

# Merge user features
ratings = ratings.merge(
    users[['user_id', 'user_id_enc', 'gender_enc', 'age_enc', 'occupation_enc', 'zip_enc']], 
    on='user_id', 
    how='left'
)

# Merge movie features
ratings = ratings.merge(
    movies[['movie_id', 'movie_id_enc', 'release_year', 'genre_ids']], 
    on='movie_id',
    how='left'
)

# Create binary label (rating >= 3 is positive)
RATING_THRESHOLD = 3
ratings['label'] = (ratings['rating'] >= RATING_THRESHOLD).astype(int)

# Time features
ratings['timestamp_dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['rating_hour'] = ratings['timestamp_dt'].dt.hour + 1
ratings['rating_weekday'] = ratings['timestamp_dt'].dt.weekday + 1
ratings['rating_month'] = ratings['timestamp_dt'].dt.month

# Year encoding (offset from base year)
BASE_YEAR = 1900
ratings['year_enc'] = (ratings['timestamp_dt'].dt.year - BASE_YEAR + 1).astype(int)
ratings['release_year_enc'] = (ratings['release_year'].fillna(BASE_YEAR) - BASE_YEAR + 1).astype(int)

# Sort by user and time for sequence generation
ratings = ratings.sort_values(['user_id_enc', 'timestamp']).reset_index(drop=True)

print(f"Total ratings after merge: {len(ratings)}")
print(f"Positive samples: {(ratings['label'] == 1).sum()}")
print(f"Negative samples: {(ratings['label'] == 0).sum()}")

# ============================================================================
# 7. GENERATE HISTORY & HARD NEGATIVES
# ============================================================================
print("\nGenerating user history and hard negatives...")

def generate_history_and_negatives(group, all_movie_ids, movie_genre_map, num_negatives=10):
    """
    For each rating, generate:
    - hist_movie_ids: Last 20 movies user interacted with
    - hard_neg_ids: Same-genre movies user hasn't seen (for positive samples)
    
    Args:
        group: User's rating history (sorted by timestamp)
        all_movie_ids: Set of all valid movie IDs
        movie_genre_map: Dict mapping movie_id_enc -> set of genre_ids
        num_negatives: Number of hard negatives per positive sample
    """
    movie_id_list = group['movie_id_enc'].tolist()
    label_list = group['label'].tolist()
    
    hist_movie = []
    hard_negatives = []
    
    for i in range(len(movie_id_list)):
        # Build history sequence
        if i == 0:
            # First rating - no history
            hist = [0] * 20
            interacted_before = set()
        else:
            # Take last 20 movies before current rating
            hist = movie_id_list[:i]
            if len(hist) > 20:
                hist = hist[-20:]  # Keep most recent 20
            else:
                hist = hist + [0] * (20 - len(hist))  # Pad to length 20
            
            interacted_before = set(movie_id_list[:i])
        
        hist_movie.append(hist)
        
        # Generate hard negatives (only for positive samples)
        if label_list[i] == 1:
            current_movie = movie_id_list[i]
            current_genres = movie_genre_map.get(current_movie, set())
            
            # Find unseen movies with overlapping genres
            same_genre_unseen = [
                mid for mid in all_movie_ids 
                if mid not in interacted_before 
                and mid != current_movie
                and len(movie_genre_map.get(mid, set()) & current_genres) > 0
            ]
            
            # Sample hard negatives
            if len(same_genre_unseen) >= num_negatives:
                negs = random.sample(same_genre_unseen, num_negatives)
            else:
                # Mix same-genre + random unseen
                negs = same_genre_unseen.copy()
                remaining = num_negatives - len(negs)
                available = list(all_movie_ids - interacted_before - {current_movie})
                
                if len(available) >= remaining:
                    negs.extend(random.sample(available, remaining))
                else:
                    # Not enough movies - pad with zeros
                    negs.extend(available)
                    negs.extend([0] * (num_negatives - len(negs)))
        else:
            # Negative samples don't need hard negatives
            negs = [0] * num_negatives
        
        hard_negatives.append(negs)
    
    group['hist_movie_ids'] = hist_movie
    group['hard_neg_ids'] = hard_negatives
    
    return group


all_movie_ids = set(ratings['movie_id_enc'].unique())

ratings = ratings.groupby('user_id_enc', group_keys=False).progress_apply(
    lambda x: generate_history_and_negatives(
        x, 
        all_movie_ids, 
        movie_genre_dict_set,  # Use set version for faster intersection
        num_negatives=10
    )
)

# ============================================================================
# 8. GENERATE HISTORY GENRE SEQUENCES
# ============================================================================
print("\nGenerating history genre features...")

def convert_hist_to_genres(movie_id_list):
    """Convert list of movie IDs to list of genre ID lists"""
    return [movie_genre_dict_padded.get(mid, [0, 0, 0]) for mid in movie_id_list]

ratings['hist_genre_ids'] = ratings['hist_movie_ids'].progress_apply(convert_hist_to_genres)

# ============================================================================
# 9. FILTER TO POSITIVE SAMPLES ONLY
# ============================================================================
print("\nFiltering to positive samples only...")
print(f"Before: {len(ratings)} samples")

ratings = ratings[ratings['label'] == 1].copy()

print(f"After: {len(ratings)} positive samples")

# ============================================================================
# 10. TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\nSplitting train/val/test...")

# Rank by timestamp within each user (most recent = 1)
ratings['rank_latest'] = ratings.groupby('user_id_enc')['timestamp'].rank(
    method='first', 
    ascending=False
)

# Split: last 2 interactions for val/test, rest for training
train = ratings[ratings['rank_latest'] > 2].copy()
val = ratings[ratings['rank_latest'] == 2].copy()
test = ratings[ratings['rank_latest'] == 1].copy()

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ============================================================================
# 11. COMPUTE STATISTICAL FEATURES (from train only)
# ============================================================================
print("\nComputing statistical features from training data...")

# User activity (how many ratings)
train_user_activity = train['user_id_enc'].value_counts()

# Movie popularity (how many times rated)
train_movie_pop = train['movie_id_enc'].value_counts()

# Movie average rating
train_movie_avg_rate = train.groupby('movie_id_enc')['rating'].mean()

def add_stat_features(df):
    """Add statistical features with log normalization"""
    df_out = df.copy()
    
    # Map features
    df_out['user_activity'] = df_out['user_id_enc'].map(train_user_activity).fillna(0)
    df_out['movie_pop'] = df_out['movie_id_enc'].map(train_movie_pop).fillna(0)
    df_out['movie_avg_rate'] = df_out['movie_id_enc'].map(train_movie_avg_rate).fillna(0)
    
    # Log transform (add 1 to avoid log(0))
    df_out['user_activity_log'] = np.log1p(df_out['user_activity'])
    df_out['movie_pop_log'] = np.log1p(df_out['movie_pop'])
    df_out['movie_avg_rate_log'] = np.log1p(df_out['movie_avg_rate'])
    
    return df_out

train = add_stat_features(train)
val = add_stat_features(val)
test = add_stat_features(test)

# ============================================================================
# 12. SAVE PROCESSED DATA
# ============================================================================
print("\nSaving processed data...")

train.to_pickle("./data/cleaned/train_set.pkl")
val.to_pickle("./data/cleaned/val_set.pkl")
test.to_pickle("./data/cleaned/test_set.pkl")
movies.to_pickle("./data/cleaned/item_set.pkl")

# Save encoders for inference
with open("./data/cleaned/encoders.pkl", "wb") as f:
    pickle.dump({
        'user_encoder': lbe_user, 
        'movie_encoder': lbe_movie,
        'zip_encoder': lbe_zip,
        'gender_encoder': lbe_gender,
        'age_encoder': lbe_age,
        'occupation_encoder': lbe_occ,
        'genre_map': movie_genre_dict_padded,
        'genre_vocab_size': len(genre2int) + 1,  # +1 for padding (0)
        'base_year': BASE_YEAR
    }, f)

# ============================================================================
# 13. VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print("\nVocabulary sizes:")
print(f"Users: {ratings['user_id_enc'].max()} (config expects 6060)")
print(f"Movies: {ratings['movie_id_enc'].max()} (config expects 3500)")
print(f"Genres: {len(genre2int) + 1} (config expects 25)")
print(f"Gender: {users['gender_enc'].max()} (config expects 3)")
print(f"Age: {users['age_enc'].max()} (config expects 9)")
print(f"Occupation: {users['occupation_enc'].max()} (config expects 22)")
print(f"Zip: {users['zip_enc'].max()} (config expects 685)")

print("\nSample train data:")
print(train[['user_id_enc', 'movie_id_enc', 'genre_ids', 'rating', 'label']].head())

print("\nHistory sequences (first user):")
sample = train[train['user_id_enc'] == 1].head(3)
for idx, row in sample.iterrows():
    print(f"\nRating #{idx}:")
    print(f"  Movie: {row['movie_id_enc']}")
    print(f"  History: {row['hist_movie_ids']}")
    print(f"  Hard Negs: {row['hard_neg_ids'][:5]}...")

print("\nMissing values check:")
print(train[['user_activity_log', 'movie_pop_log', 'movie_avg_rate_log']].isna().sum())

print("\nData saved successfully!")
print("="*80)

