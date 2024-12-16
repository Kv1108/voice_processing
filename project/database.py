# database.py
import mysql.connector
from mysql.connector import Error

# Function to establish connection to the MySQL database
def create_connection():
    """Creates and returns a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',             # Replace with your database host
            user='your_username',         # Replace with your MySQL username
            password='your_password',     # Replace with your MySQL password
            database='your_database'      # Replace with your database name
        )
        if connection.is_connected():
            print("Connection to MySQL database established successfully.")
            return connection
        else:
            print("Failed to connect to the database.")
            return None
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Function to store transcription, speaker labels, and timestamps in the database
def store_transcription(transcription, speaker_labels, timestamps):
    """Stores the transcription, speaker labels, and timestamps in the database."""
    try:
        # Establish the connection
        connection = create_connection()
        
        if connection:
            cursor = connection.cursor()

            # SQL query to insert the transcription data
            query = """
            INSERT INTO transcriptions (transcript, speakers, timestamps)
            VALUES (%s, %s, %s)
            """
            # Prepare the data for insertion
            data = (transcription, str(speaker_labels), str(timestamps))

            # Execute the query and commit the changes
            cursor.execute(query, data)
            connection.commit()

            print("Transcription and speaker data stored successfully.")
        else:
            print("No connection to database. Data was not stored.")
    
    except Error as e:
        print(f"Error while storing transcription data: {e}")
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

# Example function to fetch transcription data (if needed)
def fetch_transcription_data():
    """Fetches all transcription data from the database."""
    try:
        connection = create_connection()

        if connection:
            cursor = connection.cursor()

            # SQL query to fetch all data
            query = "SELECT * FROM transcriptions"
            cursor.execute(query)
            
            # Fetch all rows
            rows = cursor.fetchall()

            for row in rows:
                print(f"Transcript: {row[0]}, Speakers: {row[1]}, Timestamps: {row[2]}")

    except Error as e:
        print(f"Error while fetching transcription data: {e}")
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")
