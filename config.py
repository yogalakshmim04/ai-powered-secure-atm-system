
    # config.py
import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",         # your MySQL username
        password="",         # your MySQL password
        database="atm_face_db"
    )