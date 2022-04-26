"""
Author: hjian42@icloud.com

This script downloads the original user profile images
"""
import requests
import shutil
import pathlib
import pandas as pd
import os, json


def download_avatar(url, filename):
    """Download the avatar image to a given file."""
    url = url.replace("_normal", "")
    filename = filename + pathlib.Path(url).suffix
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as img:
        shutil.copyfileobj(response.raw, img)
        print("Downloaded twitter avatar to", filename)
    del response
    

# download images for the old
df = pd.read_csv("./all_users_info.csv")
print(df.head(10))

for row in df.iterrows():
    download_avatar(row[1].profile_image_url, "./pic/{}".format(row[1].screen_name))