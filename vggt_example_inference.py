import os
import requests

# Config
SERVER_URL = "http://localhost:1234/predict/"
IMAGE_DIR = "/home/wilshen/datasets/feijoa/teddy_v1/keyframes/images"  # replace with your actual path

# Gather all .jpg files
image_files = [
    os.path.join(IMAGE_DIR, fname)
    for fname in sorted(os.listdir(IMAGE_DIR))
    if fname.lower().endswith(".jpg")
]

if not image_files:
    raise ValueError("No JPG images found in the directory.")

# Prepare multipart form-data payload
file_objs = [open(f, "rb") for f in image_files]
try:
    files = [("images", (os.path.basename(f.name), f, "image/jpeg")) for f in file_objs]
    response = requests.post(SERVER_URL, files=files)
finally:
    for f in file_objs:
        f.close()

# Handle response
if response.status_code == 200:
    result = response.json()
    print("Prediction keys:", list(result.keys()))
    print("Sample depth shape:", len(result['depth']), "x", len(result['depth'][0]))
else:
    print(f"Request failed: {response.status_code}")
    print(response.text)
