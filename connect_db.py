import os
from pymongo import MongoClient
from config import config

client = MongoClient(config['DATABASE_URL'])  

db = client["bookings-app"]