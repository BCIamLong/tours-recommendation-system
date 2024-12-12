import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from preprocess_data import load_data,preprocess_tours,preprocess_posts, preprocess_interactions, save_model_data, load_model_data
from scipy.sparse import csr_matrix

model = NearestNeighbors(algorithm= 'brute')

tour_data, user_data, post_data, bookmark_data, review_data = load_data()
df_tours = preprocess_tours(tour_data)
df_posts = preprocess_posts(post_data)
interactions_df = preprocess_interactions(df_reviews=review_data, df_bookmarks=bookmark_data, df_posts=df_posts)

# Define weights for each interaction type (tweak as needed)
weights = {
    'rating': 1.0,
    'likes_count': 0.5,
    'comments_count': 0.3,
    'bookmark': 0.7,
    'review': 0.9,
    'saved_count': 0.4
}

# Step 1: Combine interaction types into a single score
interactions_df['weighted_score'] = (
    interactions_df['rating'] * weights['rating'] +
    interactions_df['likes_count'] * weights['likes_count'] +
    interactions_df['comments_count'] * weights['comments_count'] +
    interactions_df['bookmark'] * weights['bookmark'] +
    interactions_df['review'] * weights['review'] +
    interactions_df['saved_count'] * weights['saved_count']
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

# print(user_tour_matrix.shape)

# Step 2: Compute user-user similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Step 3: Predict ratings for missing values
def predict_ratings(user_index, user_similarity, user_item_matrix):
    # user_ratings = user_item_matrix.values[user_index]
    if isinstance(user_index, str):
        user_tour_matrix_df = user_tour_matrix.reset_index()
        user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_index].tolist()

        if not user_idx_list:
            raise ValueError(f"User ID {user_index} not found in interactions DataFrame.")
        user_idx = user_idx_list[0]
    else:
        user_idx = user_index
    similarities = user_similarity[user_idx]
    weighted_sum = np.dot(similarities, user_item_matrix.fillna(0))
    similarity_sum = np.abs(similarities).sum()
    predicted_ratings = weighted_sum / similarity_sum
    return predicted_ratings

# Example: Predict ratings for user 1
# user_index = "guest-8dd3c11c-1df4-4f0b-b26d-9d96d80bc28a" # Corresponds to User 1
# predicted_ratings = predict_ratings(user_index, user_similarity, user_item_matrix)
# predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.columns, columns=['Predicted Rating'])

# # Step 4: Recommend top tours for the user
# top_recommendations = predicted_ratings_df.sort_values(by='Predicted Rating', ascending=False)
# print("Top recommendations for User 1:")
# print(top_recommendations.head())

cf_predictions = np.array([predict_ratings(i, user_similarity, user_item_matrix) for i in range(len(user_item_matrix))])


# --------------------------------------------------------------------------------------------
df_tours['tourId'] = df_tours['_id']
df_tours['Features'] = df_tours['name'] + " " + df_tours['description']

vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df_tours['Features'])

# Compute item-item similarity
content_similarity = cosine_similarity(feature_matrix)
content_similarity_df = pd.DataFrame(content_similarity, index=df_tours['tourId'], columns=df_tours['tourId'])

common_tours = list(content_similarity_df.index)  # Tours in content_similarity
user_item_matrix_aligned = user_item_matrix.reindex(columns=common_tours, fill_value=0)

def predict_cbf_ratings(user_index, user_item_matrix, content_similarity):
    if isinstance(user_index, str):
        user_tour_matrix_df = user_tour_matrix.reset_index()
        user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_index].tolist()

        if not user_idx_list:
            raise ValueError(f"User ID {user_index} not found in interactions DataFrame.")
        user_idx = user_idx_list[0]
    else:
        user_idx = user_index

    user_ratings = user_item_matrix.values[user_idx]
    # print(content_similarity.shape, user_ratings.shape)
    predicted_ratings = np.dot(content_similarity, user_ratings)
    return predicted_ratings


# tour_index = "guest-8dd3c11c-1df4-4f0b-b26d-9d96d80bc28a" # Corresponds to User 1
# predicted_ratings_tours = predict_cbf_ratings(tour_index, user_item_matrix,content_similarity)
# predicted_ratings_tours_df = pd.DataFrame(predicted_ratings_tours, index=user_item_matrix.columns, columns=['Predicted Rating'])

# # Step 4: Recommend top tours for the user
# top_recommendations = predicted_ratings_tours_df.sort_values(by='Predicted Rating', ascending=False)
# print("Top recommendations for tour 1:")
# print(top_recommendations.head())

cbf_predictions = np.array([predict_cbf_ratings(i, user_item_matrix, content_similarity) for i in range(len(user_item_matrix))])


# ---------------------------------------------------------
# Step 5: Combine CF and CBF Scores
# Normalize predictions for each method
scaler = MinMaxScaler()
cf_normalized = scaler.fit_transform(cf_predictions)
cbf_normalized = scaler.fit_transform(cbf_predictions)

# Weighted combination
alpha = 0.5  # Weight for CF; 1-alpha for CBF
hybrid_scores = alpha * cf_normalized + (1 - alpha) * cbf_normalized

# Step 6: Recommend Tours for a User

user_tour_matrix_df = user_tour_matrix.reset_index()
user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == "5c8a24402f8fb814b56fa190"].tolist()

user_index = user_idx_list[0] # Example: User 1
user_hybrid_scores = hybrid_scores[user_index]
recommended_tours = pd.DataFrame({
    'Tour': user_item_matrix.columns,
    'Hybrid Score': user_hybrid_scores
}).sort_values(by='Hybrid Score', ascending=False)

print("Top recommendations for User 1:")
print(recommended_tours.head(n=21))