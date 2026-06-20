import os
import requests
from dotenv import load_dotenv


# Your OpenRouter API key (replace with your actual key)
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Our question to DeepSeek
data = {
    "model": "deepseek/deepseek-chat",
    "messages": [{"role": "user", "content": "What's the best way to learn programming?"}]
}

# Send the request
response = requests.post(API_URL, json=data, headers=headers,timeout=20)

# Check if it worked
if response.status_code == 200:
    # Extract and print just the AI's response text
    ai_message = response.json()['choices'][0]['message']['content']
    print(f"DeepSeek says: {ai_message}")
else:
    print(f"Oops! Something went wrong. Status code: {response.status_code}")
    print(f"Error details: {response.text}")
