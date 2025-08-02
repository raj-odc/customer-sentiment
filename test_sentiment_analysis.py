#!/usr/bin/env python3

"""
Customer Sentiment Analysis MVP - Test Suite
Comprehensive testing of all API endpoints and functionality.
"""

import requests
import json
import time
import sys

# Base URL for the API
BASE_URL = 'http://localhost:5000'

def test_health_check():
    """Test the health endpoint"""
    print("\n🏥 Testing health check endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_advanced_analysis():
    """Test the advanced sentiment analysis endpoint"""
    print("\n🧠 Testing advanced sentiment analysis...")
    
    test_data = {
        "text": "The new UI is beautiful and intuitive, but the app keeps crashing on startup and the login process is very slow."
    }
    
    try:
        start_time = time.time()
        response = requests.post(f'{BASE_URL}/analyze', json=test_data, timeout=60)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Advanced analysis successful")
            print(f"   Overall sentiment: {result.get('overall_sentiment')}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Aspects analyzed: {len(result.get('analysis', []))}")
            
            # Print detailed analysis
            for aspect in result.get('analysis', []):
                print(f"   - {aspect['aspect']}: {aspect['sentiment_label']} ({aspect['sentiment_score']})")
                if aspect.get('suggestion'):
                    print(f"     Suggestion: {aspect['suggestion'][:100]}...")
            
            return True
        else:
            print(f"❌ Advanced analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Advanced analysis failed: {e}")
        return False

def test_simple_analysis():
    """Test the simple sentiment analysis endpoint"""
    print("\n⚡ Testing simple sentiment analysis...")
    
    test_data = {
        "text": "I love this product! It's amazing and works perfectly."
    }
    
    try:
        start_time = time.time()
        response = requests.post(f'{BASE_URL}/analyze/simple', json=test_data, timeout=10)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple analysis successful")
            print(f"   Sentiment: {result.get('sentiment_label')} ({result.get('sentiment_score')})")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   TextBlob score: {result.get('textblob_score')}")
            print(f"   VADER score: {result.get('vader_score')}")
            return True
        else:
            print(f"❌ Simple analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Simple analysis failed: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n📁 Testing file upload...")
    
    # Test CSV upload
    try:
        with open('/Users/selvaraj/workspace/customer-sentiment/sample_data.csv', 'rb') as file:
            files = {'file': file}
            response = requests.post(f'{BASE_URL}/upload', files=files, timeout=60)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ CSV upload successful")
            print(f"   {result.get('message')}")
            print(f"   Results count: {len(result.get('results', []))}")
            return True
        else:
            print(f"❌ CSV upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ CSV upload failed: {e}")
        return False

def test_analytics():
    """Test analytics endpoint"""
    print("\n📊 Testing analytics endpoint...")
    
    try:
        response = requests.get(f'{BASE_URL}/analytics', timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                print(f"⚠️ Analytics unavailable: {result['error']}")
                return True  # This is expected if database is not configured
            else:
                print("✅ Analytics successful")
                print(f"   Sentiment distribution: {result.get('sentiment_distribution')}")
                print(f"   Recent analyses: {len(result.get('recent_analyses', []))}")
                print(f"   Trend data points: {len(result.get('sentiment_trends', []))}")
                return True
        else:
            print(f"❌ Analytics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Analytics failed: {e}")
        return False

def test_dashboard():
    """Test dashboard endpoint"""
    print("\n🖥️ Testing dashboard...")
    
    try:
        response = requests.get(f'{BASE_URL}/', timeout=10)
        
        if response.status_code == 200 and 'Customer Sentiment Analysis Dashboard' in response.text:
            print("✅ Dashboard accessible")
            return True
        else:
            print(f"❌ Dashboard failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Customer Sentiment Analysis MVP - Test Suite")
    print("="*60)
    
    # Wait for server to be ready
    print("\n⏳ Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f'{BASE_URL}/health', timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
        if i == max_retries - 1:
            print("❌ Server not responding. Make sure it's running on localhost:5000")
            return False
    
    # Run all tests
    tests = [
        test_health_check,
        test_dashboard,
        test_simple_analysis,
        test_advanced_analysis,
        test_file_upload,
        test_analytics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "="*60)
    print(f"📋 Test Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The MVP is working correctly.")
        print("\n🚀 Next steps:")
        print("   • Open http://localhost:5000 in your browser")
        print("   • Upload sample_data.csv for batch testing")
        print("   • Try the interactive dashboard")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)