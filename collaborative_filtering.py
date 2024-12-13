import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering(user_index, user_item_matrix):
    # user_ratings = user_item_matrix.values[user_index]
    if isinstance(user_index, str):
        user_tour_matrix_df = user_item_matrix.reset_index()
        user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_index].tolist()

        if not user_idx_list:
            raise ValueError(f"User ID {user_index} not found in interactions DataFrame.")
        user_idx = user_idx_list[0]
    else:
        user_idx = user_index



    user_similarity = cosine_similarity(user_item_matrix)
    # user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    similarities = user_similarity[user_idx]
    weighted_sum = np.dot(similarities, user_item_matrix.fillna(0))
    similarity_sum = np.abs(similarities).sum()
    predicted_ratings = weighted_sum / similarity_sum
    return predicted_ratings
    # return predicted_ratings, user_similarity