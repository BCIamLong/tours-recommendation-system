import pandas as pd
import traceback

from flask import Flask, request, jsonify, make_response, json
from train import train
from flask_cors import CORS
from preprocess_data import load_model_data, load_data

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://tours-booking-api.onrender.com"]}}, supports_credentials=True)

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
        hybrid_scores,user_item_matrix = load_model_data('train_data.pkl')

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


# ------------------------------------RUN FLASK APP------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=3100)