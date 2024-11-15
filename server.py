from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Replace this with the URL Deepnote will use to access the data
DEEPNOTE_API_URL = "https://deepnote-project-url.com/api-endpoint"  

# Password to check that requests come from Roblox
ACCESS_PASSWORD = "your_secure_password"

@app.route('/webhook', methods=['POST'])
def webhook():
    # Check the password
    if request.json.get("password") != ACCESS_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 403

    # Get the data Roblox sent
    data = request.json.get("data")
    
    # Send data to Deepnote or save it (for now, we'll just print it)
    print("Data from Roblox:", data)
    
    return jsonify({"status": "success", "message": "Data received"})

if __name__ == '__main__':
    app.run(debug=True)
