# Pipeline for the demographic analysis

## crawl user information

We first extract all the user information of all the covid-19 drug tweets and store them into a csv file including their `user_id`, `name`, `screen_name`, `description`, and `profile_image_url`. See the following:

```
# this script takes in a intput file `all_users.txt` and generates `all_users_info.json`. 
python get_user_info.py

# convert the json format to csv, generating `all_users_info.csv`
python convert_json2csv.py
```

## infer age, gender, and organization status with M3

```
# move the file
cp all_users_info.csv ./m3inference/coviddrug
cd ./m3inference/coviddrug

# download images using the URLs from `all_users_info.csv`
python crawl_user_images.py

# resize the images to the right format
python resize_images.py

# split users into image+text users and text-only users
# generating `users_with_images.jsonl` and `users_only_texts.jsonl`
python split_users.py

# preprocess users with M3
cd ..
python scripts/preprocess.py --source_dir coviddrug/pic_resize_400x400/ --output_dir coviddrug/pic_resized/ --jsonl_path coviddrug/users_with_images.jsonl --jsonl_outpath coviddrug/users_with_images_resized.jsonl --verbose

# make inferences for all the users
cd coviddrug
python infer_users_demographic.py
``` 

## infer political affiliation

```
# get friends of the target users, saving into `./friends_info`
cp all_users_info.csv ./political_inference/coviddrug
cd political_inference
python get_friends.py

# infer political orientation with politicians, saving predictions to `covid_user_political_follow.csv`
# check the `politician_inference.ipynb`
```


## Use the data for demographic analysis

Check the `plots/analyzing-stance-demographic.ipynb`