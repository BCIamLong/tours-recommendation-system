import pandas as pd
import traceback

from flask import Flask, request, jsonify, make_response, json
from preprocess_data import load_data,preprocess_tours,preprocess_posts, preprocess_interactions, save_model_data, load_model_data
from model import calculate_content_similarity, train_collaborative_filtering, hybrid_recommendations
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://tours-booking-api.onrender.com"]}}, supports_credentials=True)

@app.route('/train', methods=['GET'])
def train_model():
    """API route to train the model and save it to a file."""
    try:
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        df_tours = preprocess_tours(tour_data)
        df_posts = preprocess_posts(post_data)
        interactions_df = preprocess_interactions(df_reviews=review_data, df_bookmarks=bookmark_data, df_posts=df_posts)

        # Load data
        tours_df = df_tours
       
        # Train collaborative filtering
        user_tour_matrix, collaborative_sim = train_collaborative_filtering(interactions_df)
        # user_tour_matrix, collaborative_sim = cosine_similarity(interactions_df)

        # Train content-based filtering
        content_sim = calculate_content_similarity(tours_df)

        # Save the model data
        save_model_data('model_data.pkl', user_tour_matrix, collaborative_sim, content_sim)

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
        # Load the pre-trained model
        # model_data = load_model_data('model_data.pkl')
        user_tour_matrix, collaborative_sim, content_sim = load_model_data('model_data.pkl')
        tour_data, user_data, post_data, bookmark_data, review_data = load_data()
        df_tours = tour_data

        # Generate hybrid recommendations
        # user_id, user_tour_matrix, collaborative_sim, content_sim, df_tours
        recommendations = hybrid_recommendations(user_id, user_tour_matrix, collaborative_sim, content_sim, df_tours)
        # recommendations = hybrid_recommendations(user_id, model_data, df_tours)
        # print(recommendations)
        return jsonify({"count": len(recommendations), "recommendations": recommendations}), 200

    except ValueError as e:
        traceback.print_exc()
        return jsonify({"error": "Invalid user_id", "details": str(e)}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to generate recommendations", "details": str(e)}), 500


# ------------------------------------RUN FLASK APP------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=3100)