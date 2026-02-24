# Test script for the SMS Fraud API
import requests
import json

def test_api(base_url="http://localhost:5000"):
    """Test the SMS Fraud Detection API"""

    # Test single prediction
    print("Testing single prediction...")
    response = requests.post(f"{base_url}/api/predict",
                           json={"text": "Free money! Click here"},
                           headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

    # Test batch prediction
    print("\nTesting batch prediction...")
    response = requests.post(f"{base_url}/api/batch_predict",
                           json={"texts": ["Free money!", "Hello world", "Win lottery now!"]},
                           headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {len(result['results'])} predictions")
        for res in result['results']:
            print(f"  '{res['text'][:20]}...' -> Fraud: {res['is_fraud']}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    test_api(base_url)