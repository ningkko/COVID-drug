"""
Author: hjian42@icloud.com

This script resizes images to 400 x 400
"""

from PIL import Image
import glob
import pathlib
from tqdm import tqdm

for pathname in tqdm(glob.glob("pic/*")):
    try:
        image = Image.open(pathname)
        new_pathname = pathname.replace("pic", "pic_resize_400x400")
        image.thumbnail((400, 400))
        image.save(new_pathname)
    except:
        print(pathname)
        print("IMAGE CANNOT BE LOADED!")
