import requests

BOT_TOKEN = "7257965363:AAG3moxIMxrge1v49YzTQ9MrDFFYUFAzHzc"

def get_chat_id():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    print(response.json())

get_chat_id()
