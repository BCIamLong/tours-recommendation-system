import pandas as pd
import traceback

from flask import Flask, request, jsonify
from pandas import json_normalize
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Connect to MongoDB (replace with your MongoDB connection string)
# client = MongoClient("mongodb://localhost:27017/")  # Local MongoDB instance
client = MongoClient("mongodb+srv://longkai:Iy6D10VoSallC75q@cluster0.rr9xh0z.mongodb.net/bookings-app?retryWrites=true&w=majority")  
app = Flask(__name__)
# or for cloud MongoDB (replace <username>, <password>, and <cluster_url>):
# client = MongoClient("mongodb+srv://<username>:<password>@<cluster_url>")

# Select a database
db = client["bookings-app"]
# db = client["booking-app"]

# Select a collection
Tours = db["tours"]
Users = db['guests']
Posts = db['posts']
Bookmarks = db['bookmarks']
Reviews = db['reviews']

tour_data = Tours.find()
user_data = Users.find({}, {'_id': 1, 'fullName': 1, 'nationality': 1})
posts_data = Posts.find()
bookmarks_data = Bookmarks.find()
reviews_data = Reviews.find()

df_tours = pd.DataFrame(list(tour_data))
df_users = pd.DataFrame(list(user_data))
df_posts = pd.DataFrame(list(posts_data))
df_bookmarks = pd.DataFrame(list(bookmarks_data))
df_reviews = pd.DataFrame(list(reviews_data))


# ------------------------------------EDIT DATA FOR TOURS-----------------------------
# Function to flatten the 'locations' field
def flatten_locations(locations):
    return [location.get('address', 'No address') for location in locations]
    # return [location['address'] for location in locations]

# Function to flatten the 'startDates' field
def flatten_start_dates(start_dates):
    return [start_date['date'] for start_date in start_dates]

# Apply the functions to the respective fields
df_tours['locations_address'] = df_tours['locations'].apply(flatten_locations)
df_tours['startDates_dates'] = df_tours['startDates'].apply(flatten_start_dates)

# Flatten 'startLocation'
start_location_df_tours = pd.json_normalize(df_tours['startLocation'])
start_location_df_tours.columns = ['startLocation_' + col for col in start_location_df_tours.columns]

# Combine the flattened fields with the original DataFrame
df_tours = df_tours.drop(columns=['locations', 'startLocation', 'startDates'])
df_tours = pd.concat([df_tours, start_location_df_tours], axis=1)

# ------------------------------------EDIT DATA FOR POSTS-----------------------------
df_posts['likes_count'] = df_posts['likes'].apply(len)
df_posts['comments_count'] = df_posts['comments'].apply(len)
df_posts['saved_count'] = df_posts['bookmarks'].apply(len)

# Keep only the relevant fields
df_posts = df_posts[['_id', 'userId', 'tourId', 'title', 'description', 'images', 'shares', 'likes_count', 'comments_count', 'saved_count']]
df_posts.rename(columns={'_id': 'post_id'}, inplace=True)

# ------------------------------------EDIT DATA FOR BOOKMARKS-----------------------------
df_bookmarks.rename(columns={'_id': 'bookmark_id', 'user': 'userId', 'cabin': 'tourId'}, inplace=True)
df_bookmarks['bookmark'] = 1

# ------------------------------------EDIT DATA FOR REVIEWS-----------------------------
df_reviews.rename(columns={'_id': 'review_id','user': 'userId', 'cabin': 'tourId'}, inplace=True)
df_reviews['review'] = 1


# ------------------------------------CREATE INTERACTION DATA-----------------------------
interactions_df = pd.concat([df_reviews, df_bookmarks, df_posts], ignore_index=True) 
# Aggregate interactions for each user-item pair 
# interactions_df = interactions_df.groupby(['userId', 'tourId']).sum().reset_index() 
interactions_df = interactions_df.groupby(['userId', 'tourId']).agg({ 'rating': 'sum', 'bookmark': 'sum', 'review': 'sum','likes_count': 'sum', 'comments_count': 'sum' ,"saved_count": 'sum'}).reset_index()
# Fill NaN values with 0 
interactions_df.fillna(0, inplace=True)



# Create a pivot table
# user_tour_matrix = interactions_df.pivot_table(index='userId', columns='tourId', values='rating').fillna(0)
rating_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='rating', fill_value=0)
likes_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='likes_count', fill_value=0)
comments_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='comments_count', fill_value=0)
bookmarks_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='bookmark', fill_value=0)
reviews_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='review', fill_value=0)
saved_count_pivot = interactions_df.pivot_table(index='userId', columns='tourId', values='saved_count', fill_value=0)

# Merge the pivot tables
user_tour_matrix = pd.concat([rating_pivot,
likes_pivot,
comments_pivot,
bookmarks_pivot,
reviews_pivot,
saved_count_pivot], axis=1)

# Perform matrix factorization
svd = TruncatedSVD(n_components=12)
matrix_svd = svd.fit_transform(user_tour_matrix)

# Calculate cosine similarity
cosine_sim = cosine_similarity(matrix_svd)

user_tour_matrix_df = pd.DataFrame(user_tour_matrix)
user_tour_matrix_df = user_tour_matrix_df.reset_index()
# print(user_tour_matrix_df)

def recommend_tours(user_id, num_recommendations=5):
    # Map user_id to an index
    # user_idx = interactions_df.index[interactions_df['userId'] == user_id].tolist()[0]
    # user_idx = df_users.index[df_users['_id'] == user_id].tolist()[0]

    user_idx_list = user_tour_matrix_df.index[user_tour_matrix_df['userId'] == user_id].tolist()

    if not user_idx_list: 
        raise ValueError(f"User ID {user_id} not found in interactions DataFrame.")
    user_idx = user_idx_list[0]
    # print(user_idx)
    
    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the indices of the recommended tours
    tour_indices = [i[0] for i in sim_scores]
    recommended_tours = df_tours.iloc[tour_indices].to_dict(orient='records') 
    return recommended_tours


# Recommend tours for a user
# print("Recommended Tours for User 1:")
# print(recommend_tours('guest-58282d75-9521-4f22-8198-f7013eb0c0f6'))

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id parameter is required"}), 400
    try:
        recommendations = recommend_tours(user_id)
        return jsonify({"recommendations": recommendations})
    except IndexError as e:
        traceback.print_exc()
        # return jsonify({"error": "Invalid user_id"}), 400
        return jsonify({"error": "Invalid user_id", "details": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=3100)
