import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from bson import ObjectId

# ------------------------------------CONTENT-BASED FILTERING------------------------------------
def calculate_content_similarity(df_tours):
    """
    Calculate the similarity between tours based on content features using TF-IDF.
    """
    # replacing null value to empty string ''
    selected_features = ["name","description","type","difficulty","locations_address","startDates_dates"]
    for feature in selected_features:
        df_tours[feature].fillna('')

    # Combine relevant textual fields for content similarity
    # df_tours['content_features'] = df_tours['name'] + " " + df_tours['description'] + " " + \
    #                                df_tours['locations_address'].apply(lambda x: " ".join(x)) + " " + \
    #                                df_tours['startDates_dates'].apply(lambda x: " ".join(x))
    df_tours['content_features'] = df_tours['name'] + " " + df_tours['description'] + " " + \
                               df_tours['type'] + " " + df_tours['difficulty'] + " " + \
                               df_tours['locations_address'].apply(lambda x: " ".join(x)) + " " + \
                               df_tours['startDates_dates'].apply(lambda x: " ".join([str(date) for date in x])) 
                                 

    
    # Vectorize content features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_tours['content_features'])
    
    # Calculate cosine similarity between items (tours)
    content_sim = cosine_similarity(tfidf_matrix)
    return content_sim


# ------------------------------------COLLABORATIVE FILTERING FUNCTIONS------------------------------------
def train_collaborative_filtering(interactions_df):
    """
    Train collaborative filtering model using SVD on user-tours interaction data.
    Returns the user-tour interaction matrix and collaborative similarity matrix.
    """
    # Create pivot tables for user-tour interactions
    rating_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='rating', fill_value=0)
    likes_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='likes_count', fill_value=0)
    comments_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='comments_count', fill_value=0)
    bookmarks_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='bookmark', fill_value=0)
    reviews_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='review', fill_value=0)
    saved_count_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='saved_count', fill_value=0)

    # Merge all pivots into a single interaction matrix
    user_tour_matrix = pd.concat([
        rating_pivot,
        likes_pivot,
        comments_pivot,
        bookmarks_pivot,
        reviews_pivot,
        saved_count_pivot
    ], axis=1)

    # Perform matrix factorization using SVD
    svd = TruncatedSVD(n_components=12)
    matrix_svd = svd.fit_transform(user_tour_matrix)

    # Calculate collaborative similarity
    collaborative_sim = cosine_similarity(matrix_svd)

    return user_tour_matrix, collaborative_sim



# ------------------------------------HYBRID RECOMMENDER FUNCTION------------------------------------
def hybrid_recommendations(user_id, user_tour_matrix, collaborative_sim, content_sim, df_tours, num_recommendations, alpha=0.5):
    """
    Generate hybrid recommendations by combining collaborative and content-based filtering.
    - `alpha` controls the weight between collaborative (0) and content-based (1).
    """
    user_tour_matrix_df = user_tour_matrix.reset_index()
    user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_id].tolist()

    if not user_idx_list:
        raise ValueError(f"User ID {user_id} not found in interactions DataFrame.")
    user_idx = user_idx_list[0]

    # Get collaborative similarity scores
    collab_scores = list(enumerate(collaborative_sim[user_idx]))
    collab_scores = sorted(collab_scores, key=lambda x: x[1], reverse=True)

    # Aggregate collaborative scores with content-based similarity
    combined_scores = []
    for tour_idx, collab_score in collab_scores:
        combined_score = (alpha * content_sim[tour_idx, :].sum()) + ((1 - alpha) * collab_score)
        combined_scores.append((tour_idx, combined_score))

    
    # Sort by combined score
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]

    # Get the indices of the recommended tours
    tour_indices = [i[0] for i in combined_scores]
    df_tours = df_tours.drop(columns=['__v'])
    recommended_tours = df_tours.iloc[tour_indices].to_dict(orient='records')
    # Convert ObjectId to string
    for recommendation in recommended_tours:
        recommendation["_id"] = str(recommendation["_id"])
        for date in recommendation["startDates"]:
            date["_id"] = str(date["_id"])
        for location in recommendation["locations"]:
            location["_id"] = str(location["_id"])
    return recommended_tours
