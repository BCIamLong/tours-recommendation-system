import numpy as np

from content_based import content_based
from collaborative_filtering import collaborative_filtering
from preprocess_data import load_data,preprocess_tours,preprocess_posts, preprocess_interactions, save_model_data, load_model_data
from hybrid_recommends import hybrid_recommends


def train():
    tour_data, user_data, post_data, bookmark_data, review_data = load_data()
    df_tours = preprocess_tours(tour_data)
    df_posts,df_post_bookmarks,df_post_likes,df_post_comments = preprocess_posts(post_data)
    interactions_df = preprocess_interactions(df_reviews=review_data, df_bookmarks=bookmark_data,     df_posts=df_posts,df_post_bookmarks=df_post_bookmarks,df_post_likes=df_post_likes,df_post_comments=df_post_comments)

    # Define weights for each interaction type (tweak as needed)
    weights = {
    'rating': 1.0,
    'bookmark': 0.7,
    'review': 0.9,
    'bookmark_post': 0.4,
    'like_post': 0.5,
    'comment_post': 0.3,
    # 'comments_count': 0.3,
    # 'likes_count': 0.5,
    # 'saved_count': 0.4
    }

    # Step 1: Combine interaction types into a single score
    interactions_df['weighted_score'] = (
    interactions_df['rating'] * weights['rating'] +
    interactions_df['bookmark'] * weights['bookmark'] +
    interactions_df['review'] * weights['review'] +
    interactions_df['comment_post'] * weights['comment_post'] +
    interactions_df['like_post'] * weights['like_post'] +
    interactions_df['bookmark_post'] * weights['bookmark_post']
    # interactions_df['comments_count'] * weights['comments_count'] +
    # interactions_df['likes_count'] * weights['likes_count'] +
    # interactions_df['saved_count'] * weights['saved_count']
    )

    # Step 2: Create a pivot table with aggregated scores
    user_tour_matrix = interactions_df.pivot_table(
    index='userId',
    columns='tourId',
    values='weighted_score',
    fill_value=0
    )

    tour_ids = df_tours['_id'].tolist()
    user_tour_matrix = user_tour_matrix.reindex(columns=tour_ids, fill_value=0)

    # Create user-item matrix
    user_item_matrix = user_tour_matrix

    cf_predictions  = np.array([collaborative_filtering(i, user_item_matrix) for i in range(len(user_item_matrix))])

    cbf_predictions = np.array([content_based(i, user_item_matrix, df_tours=df_tours) for i in range(len(user_item_matrix))])

    hybrid_scores = hybrid_recommends(cf_predictions, cbf_predictions)

    save_model_data('train_data.pkl',  hybrid_scores, user_item_matrix)