import requests

# Your Application Credentials
CLIENT_ID = "30182_MEqHsNhzTMosn67W6z0tsBgdFGDciieiP59eKzoqCGlO1YOYJY"
CLIENT_SECRET = "jOGmayLGU1y4GFs8i2kti9jQLnzqdVLTArQdnVS1oIxFSQFqX0"
REDIRECT_URI = "http://localhost:8080"

# --- PASTE YOUR AUTH CODE HERE ---
AUTH_CODE = "PASTE_YOUR_AUTH_CODE_HERE"
# ---------------------------------

def get_access_token():
    if AUTH_CODE == "PASTE_YOUR_AUTH_CODE_HERE":
        print("❌ Error: You must paste the AUTH_CODE into the script first!")
        return

    print("🔄 Swapping Auth Code for Access Token...")
    token_url = "https://connect.spotware.com/oauth/v2/token"
    
    params = {
        "grant_type": "authorization_code",
        "code": AUTH_CODE,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    
    response = requests.get(token_url, params=params)
    
    if response.status_code == 200:
        token_data = response.json()
        print("\n✅ SUCCESS! Here is your permanent Access Token:")
        print("="*60)
        print(token_data.get("accessToken"))
        print("="*60)
        print("\nPaste this token into the live_bridge_ctrader.py script!")
    else:
        print("\n❌ Failed to get token.")
        print(f"Status Code: {response.status_code}")
        print(f"Error: {response.text}")
        print("\nNote: Auth codes expire in 60 seconds! You might need to generate a new one from your browser.")

if __name__ == "__main__":
    get_access_token()
