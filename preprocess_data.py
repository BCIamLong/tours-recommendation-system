import pandas as pd
import pickle

from connect_db import db

def load_data():
    Tours = db["tours"]
    Users = db['guests']
    Posts = db['posts']
    Bookmarks = db['bookmarks']
    Reviews = db['reviews']

    tour_data_raw =Tours.find()
    user_data_raw =Users.find({'deactivated': False}, {'_id': 1, 'fullName': 1, 'nationality': 1})
    post_data_raw =Posts.find()
    bookmark_data_raw =Bookmarks.find()
    review_data_raw =Reviews.find()

    """Load data from MongoDB collections into Pandas DataFrames."""
    tour_data = pd.DataFrame(list(tour_data_raw))
    user_data = pd.DataFrame(list(user_data_raw))
    post_data = pd.DataFrame(list(post_data_raw))
    bookmark_data = pd.DataFrame(list(bookmark_data_raw))
    review_data = pd.DataFrame(list(review_data_raw))
    return tour_data, user_data, post_data, bookmark_data, review_data


def preprocess_tours(df_tours):
    """Preprocess tour data."""
    def flatten_locations(locations):
        return [loc.get('address', 'No address') for loc in locations]

    def flatten_start_dates(start_dates):
        return [date['date'] for date in start_dates]

    df_tours['locations_address'] = df_tours['locations'].apply(flatten_locations)
    df_tours['startDates_dates'] = df_tours['startDates'].apply(flatten_start_dates)
    
    start_location_df = pd.json_normalize(df_tours['startLocation'])
    start_location_df.columns = ['startLocation_' + col for col in start_location_df.columns]
    
    df_tours = df_tours.drop(columns=['locations', 'startLocation', 'startDates', '__v'])
    df_tours = pd.concat([df_tours, start_location_df], axis=1)
    return df_tours


def preprocess_posts(df_posts):
    """Preprocess post data."""
    df_posts['likes_count'] = df_posts['likes'].apply(len)
    df_posts['comments_count'] = df_posts['comments'].apply(len)
    df_posts['saved_count'] = df_posts['bookmarks'].apply(len)

    # Keep only the relevant fields
    df_posts = df_posts[['_id', 'userId', 'tourId', 'title', 'description', 'images', 'shares', 'likes_count', 'comments_count', 'saved_count']]
    # df_posts = df_posts.loc[:, ['_id', 'userId', 'tourId', 'title', 'description', 'images', 'shares', 'likes_count', 'comments_count', 'saved_count']]
    df_posts.rename(columns={'_id': 'post_id'}, inplace=True)   
    # print(df_posts.dtypes) 
    return df_posts


def preprocess_interactions(df_reviews, df_bookmarks, df_posts):
    """Create and aggregate user-tour interactions."""
    df_bookmarks.rename(columns={'_id': 'bookmark_id', 'user': 'userId', 'cabin': 'tourId'}, inplace=True)
    df_bookmarks['bookmark'] = 1
    
    df_reviews.rename(columns={'_id': 'review_id', 'user': 'userId', 'cabin': 'tourId'}, inplace=True)
    df_reviews['review'] = 1
    
    interactions = pd.concat([df_reviews, df_bookmarks, df_posts], ignore_index=True)
    interactions = interactions.groupby(['userId', 'tourId']).agg({
        'rating': 'sum',
        'bookmark': 'sum',
        'review': 'sum',
        'likes_count': 'sum',
        'comments_count': 'sum',
        'saved_count': 'sum'
    }).reset_index().fillna(0)
    # print(interactions.dtypes) 
    # interactions['userId'] = interactions['userId'].astype('category').cat.codes 
    # interactions['tourId'] = interactions['tourId'].astype('category').cat.codes
    return interactions





def save_model_data(file_path, user_tour_matrix, collaborative_sim, content_sim):
    """Save trained matrices and other model data to a file."""
    with open(file_path, 'wb') as file:
        pickle.dump({
            "user_tour_matrix": user_tour_matrix,
            "collaborative_sim": collaborative_sim,
            "content_sim": content_sim
        }, file)
    print(f"Model data saved to {file_path}")


def load_model_data(file_path):
    """Load pre-trained model data from a file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Model data loaded from {file_path}")
    return data['user_tour_matrix'], data['collaborative_sim'], data['content_sim']
    # return data