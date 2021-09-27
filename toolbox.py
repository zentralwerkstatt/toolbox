import requests
import PIL.Image
from io import BytesIO
import time
import os

# Returns an image from an URL as a PIL image
def img_from_url(url, max_tries=10):
    tries = 0
    while tries < max_tries:
        try:
            response = requests.get(url, timeout=30)
            img_bytes = BytesIO(response.content)
            img = PIL.Image.open(img_bytes).convert("RGB")
            return img
        except:
            tries += 1
        time.sleep(1)

def load_img(path):
    return PIL.Image.open(path).convert("RGB")

def new_dir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder