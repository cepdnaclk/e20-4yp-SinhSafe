import requests

def test_google_api():
    word = "kohomada pako phone ගෙදර 123"
    print(f"Testing Google API with word: '{word}'...")
    
    url = "https://inputtools.google.com/request?text=" + word + "&itc=si-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # The result is usually nested like: ['SUCCESS', [['kohomada', ['කොහොමද']]]]
            result = data[1][0][1][0]
            print(f"✅ Success! API returned: {result}")
        else:
            print(f"❌ Failed. Status Code: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_google_api()