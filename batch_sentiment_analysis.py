import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json

# PostgreSQL connection config
db_config = {
    "host": "localhost",
    "port": 5432,
    "dbname": "reviews_db",
    "user": "review_user",
    "password": "review123"
}

def fetch_all_reviews():
    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, text FROM google_reviews")  # Replace reviews_table and review_text with your actual table and column names
            return cur.fetchall()
    finally:
        conn.close()

def analyze_review_text(text):
    url = 'http://localhost:5000/analyze'
    payload = {"text": text}
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.json()

def main():
    reviews = fetch_all_reviews()
    for review in reviews:
        print(f"Processing review ID: {review['id']}")
        try:
            analysis_result = analyze_review_text(review['text'])
            print(json.dumps(analysis_result, indent=2))
        except Exception as e:
            print(f"Error processing review ID {review['id']}: {e}")

if __name__ == "__main__":
    main()