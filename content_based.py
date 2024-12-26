import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def content_based(user_index, user_item_matrix, df_tours):
    if isinstance(user_index, str):
        user_tour_matrix_df = user_item_matrix.reset_index()
        user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_index].tolist()

        if not user_idx_list:
            raise ValueError(f"User ID {user_index} not found in interactions DataFrame.")
        user_idx = user_idx_list[0]
    else:
        user_idx = user_index
    
    df_tours['tourId'] = df_tours['_id']
    df_tours['Features'] = df_tours['name'] + " " + df_tours['description'] + " " + \
                               df_tours['type'] + " " + df_tours['difficulty'] + " " + \
                               df_tours['locations_address'].apply(lambda x: " ".join(x)) + " " + \
                               df_tours['startDates_dates'].apply(lambda x: " ".join([str(date) for date in x])) 

    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df_tours['Features'])

    # Compute item-item similarity
    content_similarity = cosine_similarity(feature_matrix)
    # content_similarity_df = pd.DataFrame(content_similarity, index=df_tours['tourId'], columns=df_tours['tourId'])

    user_ratings = user_item_matrix.values[user_idx]
    # print(content_similarity.shape, user_ratings.shape)
    predicted_ratings = np.dot(content_similarity, user_ratings)
    return predicted_ratings
    # return predicted_ratings,content_similarity


def content_based_for_tours(df_tours, search_str=''):
    if search_str != '':
        df_tours['Features'] = search_str
    else:
        df_tours['Features'] = df_tours['name'] + " " + df_tours['description'] + " " + \
                               df_tours['type'] + " " + df_tours['difficulty'] + " " + \
                               df_tours['locations_address'].apply(lambda x: " ".join(x)) + " " + \
                               df_tours['startDates_dates'].apply(lambda x: " ".join([str(date) for date in x])) 

    # Vectorize the Features using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df_tours['Features'])

    # Compute the similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix



def search_tours(search_query, df_tours, top_n=5):
    # Combine relevant fields into a "Features" column
    df_tours['Features'] = (
        df_tours['name'] + " " + 
        df_tours['description'] + " " +
        df_tours['type'] + " " +
        df_tours['difficulty'] + " " +
        df_tours['locations_address'].apply(lambda x: " ".join(x)) + " " +
        df_tours['startDates_dates'].apply(lambda x: " ".join([str(date) for date in x]))
    )
    
    # Add the search query as a new row in the Features column
    all_data = df_tours['Features'].tolist() + [search_query]
    
    # Vectorize the Features column
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(all_data)
    
    # Compute similarity between the search query and all tours
    similarity_scores = cosine_similarity(feature_matrix[-1], feature_matrix[:-1]).flatten()
    
    # Rank tours by similarity scores
    df_tours['Similarity Score'] = similarity_scores
    matching_tours = df_tours.sort_values(by='Similarity Score', ascending=False).head(top_n)
    
    return matching_tours, matching_tours[['name']]


def search_posts(search_query, df_posts, top_n=5):
    # Combine relevant fields into a "Features" column
    df_posts['Features'] = (
        df_posts['title'] + " " + 
        df_posts['description'] 
    )
    
    # Add the search query as a new row in the Features column
    all_data = df_posts['Features'].tolist() + [search_query]
    
    # Vectorize the Features column
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(all_data)
    
    # Compute similarity between the search query and all tours
    similarity_scores = cosine_similarity(feature_matrix[-1], feature_matrix[:-1]).flatten()
    
    # Rank tours by similarity scores
    df_posts['Similarity Score'] = similarity_scores
    matching_posts = df_posts.sort_values(by='Similarity Score', ascending=False).head(top_n)
    
    return matching_posts[['_id', 'title']]