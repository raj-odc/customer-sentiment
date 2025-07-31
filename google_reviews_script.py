import hashlib
import logging
from datetime import datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from serpapi import GoogleSearch  # Correct import for SerpApi client
from sentence_transformers import SentenceTransformer
from sqlalchemy import (Column, DateTime, Integer, String, Text, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERPAPI_API_KEY = "018aee3a954d462a80ccbbbf4d0edc33b0c0f9b96a25686d3f1416eb0ec17d5b"  # Replace with your SerpApi key
DATABASE_URL = "postgresql://review_user:review123@localhost:5432/reviews_db"

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
Base = declarative_base()


class GoogleReview(Base):
    __tablename__ = 'google_reviews'

    id = Column(Integer, primary_key=True)
    review_id = Column(String(64), unique=True, nullable=False)  # SHA256 hash
    place_id = Column(String, nullable=False)
    place_name = Column(String)
    author_name = Column(String)
    author_url = Column(String)
    profile_photo_url = Column(String)
    rating = Column(Integer)
    relative_time_description = Column(String)
    text = Column(Text)
    time = Column(DateTime)
    language = Column(String)
    original_language = Column(String)
    translated = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def generate_review_id(review):
    unique_string = f"{review.get('author_name', '')}_{review.get('time', '')}_{review.get('text', '')}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def setup_database():
    """Setup PostgreSQL database with pgvector extension and tables"""
    try:
        Base.metadata.create_all(bind=engine)

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Assuming pgvector extension is already created manually
        # cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_embeddings (
                id SERIAL PRIMARY KEY,
                review_id INTEGER REFERENCES google_reviews(id),
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS review_embeddings_idx 
            ON review_embeddings USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)

        conn.commit()
        cur.close()
        conn.close()

        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


class SerpApiReviewScraper:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_place_id(self, place_name):
        params = {
            "engine": "google_maps",
            "q": place_name,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        #logger.info(f"Place search results for '{place_name}': {results}")

        if "place_results" in results and isinstance(results["place_results"], dict):
            place = results["place_results"]
            return place.get("place_id")

        logger.warning(f"Place ID not found for {place_name}")
        return None

    def scrape_reviews(self, place_name):
        place_id = self.get_place_id(place_name)
        if not place_id:
            logger.warning(f"Cannot scrape reviews without place_id for {place_name}")
            return []

        params = {
            "engine": "google_maps_reviews",
            "place_id": place_id,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us",
            # Do NOT include 'num' here on the first request
        }

        all_reviews = []
        while True:
            search = GoogleSearch(params)
            results = search.get_dict()
            logger.info(f"SerpApi reviews results: {results}")

            reviews = results.get("reviews", [])
            if not reviews:
                logger.warning("No reviews found for this place")
                break

            for review in reviews:
                try:
                    review_time = datetime.strptime(review.get("date", ""), "%b %d, %Y")
                except Exception:
                    review_time = None

                review_data = {
                    'review_id': generate_review_id({
                        'author_name': review.get('user_name'),
                        'time': review_time,
                        'text': review.get('snippet')
                    }),
                    'place_id': place_id,
                    'place_name': place_name,
                    'author_name': review.get('user_name'),
                    'author_url': review.get('user_url'),
                    'profile_photo_url': review.get('profile_photo'),
                    'rating': int(review.get('rating', 0)),
                    'relative_time_description': review.get('relative_time_description'),
                    'text': review.get('snippet'),
                    'time': review_time,
                    'language': 'en',
                    'original_language': None,
                    'translated': None
                }
                all_reviews.append(review_data)

            next_page_token = results.get("serpapi_pagination", {}).get("next_page_token")
            logger.info(f"Next page token: {next_page_token}")
            if not next_page_token:
                break

            params["next_page_token"] = next_page_token
            params["num"] = 10
            0

        logger.info(f"Scraped total {len(all_reviews)} reviews from SerpApi")
        return all_reviews


def store_reviews_in_db(reviews):
    session = SessionLocal()
    stored_reviews = []
    try:
        for review_data in reviews:
            existing = session.query(GoogleReview).filter(
                GoogleReview.review_id == review_data['review_id']
            ).first()
            if existing:
                continue
            review = GoogleReview(**review_data)
            session.add(review)
            session.flush()
            stored_reviews.append(review)
        session.commit()

        generate_and_store_embeddings(stored_reviews, session)

        return stored_reviews
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing reviews: {e}")
        raise
    finally:
        session.close()


def generate_and_store_embeddings(reviews, session):
    conn = session.connection().connection
    cur = conn.cursor()
    try:
        for review in reviews:
            if not review.text:
                continue
            embedding = embed_model.encode(review.text)
            embedding_list = embedding.tolist()
            cur.execute("""
                INSERT INTO review_embeddings (review_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
            """, (review.id, embedding_list))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error generating embeddings: {e}")
        raise
    finally:
        cur.close()


def semantic_search_reviews(query_text, limit=5):
    query_embedding = embed_model.encode(query_text).tolist()
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT 
                gr.id,
                gr.place_name,
                gr.author_name,
                gr.rating,
                gr.text,
                gr.relative_time_description,
                1 - (re.embedding <=> %s::vector) as similarity
            FROM google_reviews gr
            JOIN review_embeddings re ON gr.id = re.review_id
            ORDER BY re.embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        results = cur.fetchall()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def main():
    setup_database()

    scraper = SerpApiReviewScraper(SERPAPI_API_KEY)

    query = "91Springboard Indiranagar Bangalore"
    logger.info(f"Scraping reviews for: {query}")
    reviews = scraper.scrape_reviews(query)

    if reviews:
        stored_reviews = store_reviews_in_db(reviews)

        search_query = "food quality and service"
        logger.info(f"Searching for: {search_query}")
        search_results = semantic_search_reviews(search_query)

        print("\nSemantic Search Results:")
        for result in search_results:
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Author: {result['author_name']}")
            print(f"Rating: {result['rating']}")
            print(f"Review: {result['text'][:200]}...")
            print("-" * 50)
    else:
        logger.warning("No reviews found to process")


if __name__ == "__main__":
    main()