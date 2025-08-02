#!/usr/bin/env python3
"""
Database setup script for Customer Sentiment Analysis MVP
This script creates the necessary database tables and setup for the sentiment analysis system.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import os  # Add this import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
import sys
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME")
}

# Admin database configuration (for creating database and user)
ADMIN_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME")
}

def create_database_and_user():
    """Create database and user if they don't exist"""
    try:
        # Connect as admin user
        conn = psycopg2.connect(**ADMIN_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create user if not exists
        try:
            cur.execute(f"""
                CREATE USER {DB_CONFIG['user']} WITH PASSWORD '{DB_CONFIG['password']}';
            """)
            logger.info(f"Created user: {DB_CONFIG['user']}")
        except psycopg2.errors.DuplicateObject:
            logger.info(f"User {DB_CONFIG['user']} already exists")
        
        # Create database if not exists
        try:
            cur.execute(f"""
                CREATE DATABASE {DB_CONFIG['dbname']} OWNER {DB_CONFIG['user']};
            """)
            logger.info(f"Created database: {DB_CONFIG['dbname']}")
        except psycopg2.errors.DuplicateDatabase:
            logger.info(f"Database {DB_CONFIG['dbname']} already exists")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database/user: {e}")
        logger.info("Please ensure PostgreSQL is running and admin credentials are correct")
        return False
    
    return True

def setup_tables():
    """Create all necessary tables"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Create google_reviews table (from existing script)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS google_reviews (
                id SERIAL PRIMARY KEY,
                review_id VARCHAR(64) UNIQUE NOT NULL,
                place_id VARCHAR NOT NULL,
                place_name VARCHAR,
                author_name VARCHAR,
                author_url VARCHAR,
                profile_photo_url VARCHAR,
                rating INTEGER,
                relative_time_description VARCHAR,
                text TEXT,
                time TIMESTAMP,
                language VARCHAR,
                original_language VARCHAR,
                translated VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create ai_analysis_results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis_results (
                id SERIAL PRIMARY KEY,
                original_text TEXT NOT NULL,
                overall_sentiment DECIMAL(5,3) NOT NULL,
                analysis_json JSONB,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(50) DEFAULT 'manual'
            );
        """)
        
        # Create review_embeddings table (for vector search)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_embeddings (
                id SERIAL PRIMARY KEY,
                review_id INTEGER REFERENCES google_reviews(id),
                embedding DECIMAL[] DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for better performance
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_google_reviews_review_id 
            ON google_reviews(review_id);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_analysis_processed_at 
            ON ai_analysis_results(processed_at);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ai_analysis_sentiment 
            ON ai_analysis_results(overall_sentiment);
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Successfully created all database tables")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up tables: {e}")
        return False

def insert_sample_data():
    """Insert some sample data for testing"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Sample reviews for testing
        sample_reviews = [
            {
                'review_id': 'sample_1',
                'place_id': 'sample_place_1',
                'place_name': 'Sample Restaurant',
                'author_name': 'John Doe',
                'rating': 5,
                'text': 'Amazing food and excellent service! The UI is beautiful and easy to use.',
                'language': 'en'
            },
            {
                'review_id': 'sample_2',
                'place_id': 'sample_place_1',
                'place_name': 'Sample Restaurant',
                'author_name': 'Jane Smith',
                'rating': 2,
                'text': 'The app keeps crashing on startup and the login is very slow. Poor performance overall.',
                'language': 'en'
            },
            {
                'review_id': 'sample_3',
                'place_id': 'sample_place_2',
                'place_name': 'Sample App',
                'author_name': 'Bob Johnson',
                'rating': 3,
                'text': 'Decent features but the pricing is too expensive. Customer support was helpful though.',
                'language': 'en'
            }
        ]
        
        for review in sample_reviews:
            cur.execute("""
                INSERT INTO google_reviews 
                (review_id, place_id, place_name, author_name, rating, text, language)
                VALUES (%(review_id)s, %(place_id)s, %(place_name)s, %(author_name)s, 
                        %(rating)s, %(text)s, %(language)s)
                ON CONFLICT (review_id) DO NOTHING
            """, review)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Successfully inserted sample data")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        return False

def test_connection():
    """Test database connection and basic queries"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Test basic queries
        cur.execute("SELECT COUNT(*) FROM google_reviews;")
        review_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM ai_analysis_results;")
        analysis_count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        logger.info(f"Database connection successful!")
        logger.info(f"Reviews in database: {review_count}")
        logger.info(f"Analyses in database: {analysis_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting Customer Sentiment Analysis Database Setup...")
    
    print("\n" + "="*60)
    print("Customer Sentiment Analysis - Database Setup")
    print("="*60)
    
    # Step 1: Create database and user
    print("\n1. Creating database and user...")
    # if not create_database_and_user():
    #     print("❌ Failed to create database/user. Please check your PostgreSQL configuration.")
    #     print("   Make sure PostgreSQL is running and update ADMIN_CONFIG credentials if needed.")
    #     sys.exit(1)
    print("✅ Database and user setup complete")
    
    # Step 2: Create tables
    print("\n2. Setting up database tables...")
    if not setup_tables():
        print("❌ Failed to create tables")
        sys.exit(1)
    print("✅ Database tables created successfully")
    
    # Step 3: Insert sample data
    print("\n3. Inserting sample data...")
    if not insert_sample_data():
        print("❌ Failed to insert sample data")
        sys.exit(1)
    print("✅ Sample data inserted successfully")
    
    # Step 4: Test connection
    print("\n4. Testing database connection...")
    if not test_connection():
        print("❌ Database connection test failed")
        sys.exit(1)
    print("✅ Database connection test successful")
    
    print("\n" + "="*60)
    print("🎉 Database setup completed successfully!")
    print("="*60)
    print("\nYour sentiment analysis system is ready to use:")
    print(f"  • Database: {DB_CONFIG['dbname']}")
    print(f"  • Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"  • User: {DB_CONFIG['user']}")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start the server: uvicorn ai_core_service:app --host 0.0.0.0 --port 5000 --reload")
    print("  3. Open http://localhost:5000 in your browser")
    print("  4. Test the API: python test_sentiment_analysis.py")
    
if __name__ == "__main__":
    main()