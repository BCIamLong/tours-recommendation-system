import pandas as pd
import traceback
import os

from flask import Flask, request, jsonify, make_response, json
from train import train
from flask_cors import CORS
from preprocess_data import load_model_data, load_data, preprocess_tours, preprocess_posts
from config import config
from content_based import content_based_for_tours, search_tours, search_posts

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173",config['CLIENT_ORIGIN'], config['SERVER_ORIGIN']]}}, supports_credentials=True)

@app.route('/train', methods=['GET'])
def train_model():
    try:
        train()
        return jsonify({"message": "Model training completed and saved successfully."}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to train the model", "details": str(e)}), 500


# ------------------------------------RECOMMENDATION ROUTE------------------------------------

@app.route('/recommend', methods=['GET'])
def recommend():
    """API route to get hybrid recommendations for a user."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id parameter is required"}), 400

    try:
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        hybrid_scores,user_item_matrix,tour_similarities_matrix = load_model_data('train_data.pkl')

        user_tour_matrix_df = user_item_matrix.reset_index()
        user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_id].tolist()

        user_index = user_idx_list[0] # Example: User 1
        user_hybrid_scores = hybrid_scores[user_index]
        recommended_tours = pd.DataFrame({
            'Tour': user_item_matrix.columns,
            'Hybrid Score': user_hybrid_scores
        }).sort_values(by='Hybrid Score', ascending=False)
        
        # print("Top recommendations for User 1:")
        # print(recommended_tours.head(n=21))
        tour_ids = recommended_tours['Tour']
        filtered_tours = tour_data[tour_data['_id'].isin(tour_ids)]
        ordered_tours = filtered_tours.set_index('_id').loc[tour_ids].reset_index()
        ordered_tours = ordered_tours.drop(columns=['__v'])

        final_recommendation_tours_list = ordered_tours.to_dict(orient='records')
        for recommendation in final_recommendation_tours_list:
            recommendation["_id"] = str(recommendation["_id"])
            for date in recommendation["startDates"]:
                date["_id"] = str(date["_id"])
            for location in recommendation["locations"]:
                location["_id"] = str(location["_id"])


        return jsonify({"count": len(ordered_tours), "recommendations": final_recommendation_tours_list, "recommendations_core": final_recommendation_tours_list}), 200

    except ValueError as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid user_id", "details": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate recommendations", "details": str(e)}), 500



@app.route('/recommend-tours', methods=['GET'])
def recommend_tours():
    try:
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        df_tours = preprocess_tours(tour_data)
        tour_index = request.args.get('tour_id')
        tour_indices = pd.Series(df_tours.index, index=df_tours['_id'])

        # Find the index of the target tour
        if tour_index not in tour_indices:
            raise ValueError(f"Tour ID {tour_index} not found in the dataset.")
        tour_index = tour_indices[tour_index]

        hybrid_scores,user_item_matrix,tour_similarities_matrix = load_model_data('train_data.pkl')
        similarity_matrix = tour_similarities_matrix
        tour_indices = pd.Series(df_tours.index, index=df_tours['_id'])

        # Find the index of the target tour

        # Retrieve similarity scores for the target tour
        similarity_scores = list(enumerate(similarity_matrix[tour_index]))

        # Sort by similarity score in descending order, excluding the target tour itself
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_matches = similarity_scores[1:5]

        # Retrieve the tour details for the top matches
        similar_tours = df_tours.iloc[[match[0] for match in top_matches]].copy()
        similar_tours['Similarity Score'] = [match[1] for match in top_matches]

        # tour_ids =  similar_tours[['_id']]
        # tours_recommendations = df_tours[df_tours['_id'].isin(tour_ids)]
        final_tours_recommendations = similar_tours.to_dict(orient='records')
        return jsonify({"count": len(final_tours_recommendations), "recommendations": final_tours_recommendations}), 200

    except ValueError as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid tour_id", "details": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate tour recommendations", "details": str(e)}), 500




@app.route('/recommend-search-tours', methods=['GET'])
def recommend_search_tours():
    try:
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        df_tours = preprocess_tours(tour_data)
        search_str = request.args.get('search_str')

        tours_recommendations, tours_recommendation_names = search_tours(df_tours=df_tours, search_query=search_str,top_n=6)

        # tour_ids =  similar_tours[['_id']]
        # tours_recommendations = df_tours[df_tours['_id'].isin(tour_ids)]
        final_tours_recommendations = tours_recommendations.to_dict(orient='records')
        final_tours_recommendation_names = tours_recommendation_names.to_dict(orient='records')
        return jsonify({"count": len(final_tours_recommendations), "recommendations": final_tours_recommendations, "recommendation_names": final_tours_recommendation_names}), 200

    except ValueError as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid tour_id", "details": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate tour recommendations", "details": str(e)}), 500
    
@app.route('/recommend-search-posts', methods=['GET'])
def recommend_search_posts():
    try:
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        df_posts = post_data
        search_str = request.args.get('search_str')

        posts_recommendations = search_posts(df_posts=df_posts, search_query=search_str,top_n=3)

        # tour_ids =  similar_tours[['_id']]
        # tours_recommendations = df_tours[df_tours['_id'].isin(tour_ids)]
        final_posts_recommendations = posts_recommendations.to_dict(orient='records')
        return jsonify({"count": len(final_posts_recommendations), "recommendations": final_posts_recommendations}), 200

    except ValueError as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid tour_id", "details": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate tour recommendations", "details": str(e)}), 500

# ------------------------------------RUN FLASK APP------------------------------------

if __name__ == '__main__':
    app.run()
    # app.run(debug=True, port=3100)