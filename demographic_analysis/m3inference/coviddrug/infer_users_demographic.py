"""
Author: hjian42@icloud.com

This script infers age, gender, and org status of users (both image-only and image+text)
"""

from m3inference import M3Inference
import pprint
import json

# image users
m3 = M3Inference(use_full_model=True, use_cuda=True, parallel=False) # see docstring for details
pred = m3.infer('./users_with_images.jsonl', batch_size=4)

with open("image_user_demographic.jsonl", "w") as out:
    for key, value in pred.items():
        jsonobj = {key: value}
        json.dump(jsonobj, out)
        out.write("\n")

# text users
m3 = M3Inference(use_full_model=False, use_cuda=True, parallel=False) # see docstring for details
pred = m3.infer('./users_only_texts.jsonl', batch_size=4)

with open("text_user_demographic.jsonl", "w") as out:
    for key, value in pred.items():
        jsonobj = {key: value}
        json.dump(jsonobj, out)
        out.write("\n")
