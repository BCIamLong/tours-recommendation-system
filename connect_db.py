from pymongo import MongoClient

client = MongoClient("mongodb+srv://longkai:Iy6D10VoSallC75q@cluster0.rr9xh0z.mongodb.net/bookings-app?retryWrites=true&w=majority")  

db = client["bookings-app"]