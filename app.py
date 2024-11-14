import pandas as pd
from pandas import json_normalize
from pymongo import MongoClient

# Connect to MongoDB (replace with your MongoDB connection string)
client = MongoClient("mongodb://localhost:27017/")  # Local MongoDB instance
# or for cloud MongoDB (replace <username>, <password>, and <cluster_url>):
# client = MongoClient("mongodb+srv://<username>:<password>@<cluster_url>")

# Select a database
db = client["booking-app"]

# Select a collection
Tours = db["tours"]
# collection.head()
print("Connected to MongoDB")

tour_data = Tours.find()


# Convert to DataFrame
df = pd.DataFrame(list(tour_data))

# Function to flatten the 'locations' field
def flatten_locations(locations):
    return [location['address'] for location in locations]

# Function to flatten the 'startDates' field
def flatten_start_dates(start_dates):
    return [start_date['date'] for start_date in start_dates]

# Apply the functions to the respective fields
df['locations_address'] = df['locations'].apply(flatten_locations)
df['startDates_dates'] = df['startDates'].apply(flatten_start_dates)

# Flatten 'startLocation'
start_location_df = pd.json_normalize(df['startLocation'])
start_location_df.columns = ['startLocation_' + col for col in start_location_df.columns]

# Combine the flattened fields with the original DataFrame
df = df.drop(columns=['locations', 'startLocation', 'startDates'])
df = pd.concat([df, start_location_df], axis=1)

# Display the DataFrame
# print(df['locations_address'])
print(df)


