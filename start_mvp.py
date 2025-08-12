#!/usr/bin/env python3
"""
Customer Sentiment Analysis MVP Startup Script
This script provides different ways to start the sentiment analysis system.
"""

from dotenv import load_dotenv
load_dotenv()  # Add this line to load environment variables

import subprocess
import sys
import os
import time
import requests
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import transformers
        import fastapi
        import psycopg2
        import pandas
        import textblob
        import vaderSentiment
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False

def check_postgres():
    """Check if PostgreSQL is running and accessible"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host = os.getenv("DB_HOST"),
            port = int(os.getenv("DB_PORT")),
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD"),
            dbname = os.getenv("DB_NAME")
        )
        conn.close()
        logger.info("✅ PostgreSQL database is accessible")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        logger.info("Run database setup: python setup_database.py")
        return False

def start_development_server():
    """Start the development server with hot reload"""
    logger.info("🚀 Starting development server...")
    try:
        subprocess.run([
            "uvicorn", "ai_core_service:app", 
            "--host", "0.0.0.0", 
            "--port", "5001", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start server: {e}")

def start_production_server():
    """Start the production server"""
    logger.info("🚀 Starting production server...")
    try:
        subprocess.run([
            "uvicorn", "ai_core_service:app", 
            "--host", "0.0.0.0", 
            "--port", "5000",
            "--workers", "4"
        ], check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start server: {e}")

def start_with_docker():
    """Start using Docker Compose"""
    logger.info("🐳 Starting with Docker Compose...")
    try:
        subprocess.run(["docker-compose", "up", "--build"], check=True)
    except KeyboardInterrupt:
        logger.info("Stopping Docker containers...")
        subprocess.run(["docker-compose", "down"])
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker startup failed: {e}")

def run_tests():
    """Run basic functionality tests"""
    logger.info("🧪 Running functionality tests...")
    
    # Wait for server to be ready
    server_url = "http://localhost:5000"
    max_retries = 30
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is ready")
                break
        except:
            pass
        
        if i == max_retries - 1:
            logger.error("❌ Server failed to start within timeout")
            return False
        
        time.sleep(1)
    
    # Test basic sentiment analysis
    try:
        test_data = {
            "text": "The app is amazing and easy to use, but it crashes sometimes and login is slow."
        }
        response = requests.post(f"{server_url}/analyze", json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Sentiment analysis test passed")
            logger.info(f"   Overall sentiment: {result.get('overall_sentiment')}")
            logger.info(f"   Aspects found: {len(result.get('analysis', []))}")
        else:
            logger.error(f"❌ Sentiment analysis test failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    # Test simple analysis
    try:
        response = requests.post(f"{server_url}/analyze/simple", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Simple analysis test passed")
            logger.info(f"   Sentiment: {result.get('sentiment_label')} ({result.get('sentiment_score')})")
        else:
            logger.error(f"❌ Simple analysis test failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Simple analysis test failed: {e}")
    
    return True

def setup_database():
    """Run database setup"""
    logger.info("🗄️ Setting up database...")
    try:
        subprocess.run([sys.executable, "setup_database.py"], check=True)
        logger.info("✅ Database setup completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def print_usage_info():
    """Print usage information"""
    print("\n" + "="*60)
    print("🎉 Customer Sentiment Analysis MVP is Ready!")
    print("="*60)
    print("\n📊 Dashboard: http://localhost:5000")
    print("📋 API Documentation: http://localhost:5000/docs")
    print("❤️ Health Check: http://localhost:5000/health")
    print("\n🔧 API Endpoints:")
    print("  POST /analyze        - Full sentiment analysis")
    print("  POST /analyze/simple - Quick sentiment analysis")
    print("  POST /upload         - Batch file upload")
    print("  GET  /analytics      - Get analytics data")
    print("  GET  /export         - Export all results")
    print("\n🧪 Test the API:")
    print("  python test_sentiment_analysis.py")
    print("\n🔄 Process batch data:")
    print("  python batch_sentiment_analysis.py")

def main():
    parser = argparse.ArgumentParser(description="Customer Sentiment Analysis MVP Startup")
    parser.add_argument("--mode", choices=["dev", "prod", "docker", "test", "setup"], 
                       default="dev", help="Startup mode")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip dependency and database checks")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Customer Sentiment Analysis MVP")
    print("="*60)
    
    # Run checks unless skipped
    if not args.skip_checks:
        print("\n🔍 Running system checks...")
        
        if not check_dependencies():
            print("\n❌ Dependency check failed. Install requirements first:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
        
        if args.mode != "docker" and args.mode != "setup":
            if not check_postgres():
                print("\n❌ Database check failed. Set up database first:")
                print("   python start_mvp.py --mode setup")
                sys.exit(1)
    
    # Execute based on mode
    if args.mode == "setup":
        if setup_database():
            print("\n✅ Database setup completed successfully!")
            print("   Now run: python start_mvp.py --mode dev")
        else:
            sys.exit(1)
    
    elif args.mode == "dev":
        print_usage_info()
        start_development_server()
    
    elif args.mode == "prod":
        print_usage_info()
        start_production_server()
    
    elif args.mode == "docker":
        print_usage_info()
        start_with_docker()
    
    elif args.mode == "test":
        print("\n🧪 Starting test server and running tests...")
        # Start server in background for testing
        import threading
        server_thread = threading.Thread(target=start_development_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a bit for server to start, then run tests
        time.sleep(5)
        run_tests()

if __name__ == "__main__":
    main()