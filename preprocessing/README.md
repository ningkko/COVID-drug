## Preprocessing
This folder contains code for downloading Tweet IDs, hydrating, and preprocessing.

1. Go to download/
   
    Follow the Retriever notebook to retrieve COVID-19-related Tweet IDs and hydrate tweets from the Twitter API. Before that you have to request a developer's account from Twitter and get keys and tokens for downloading.

2. Go to extract_drugtweets/
   
    We extract tweets with drug strings in this step. Tweets containing more than 1 string will be kept in this step for trend analysis. You'll see a tweet_amount.log file generated. The extraction process takes time, about a day on a 48-core CPU. Also it halts some time. Keep an eye on it.

3. Go to find_geo/
   
    Run find_geo1.py and find_geo2.py and then merge the results using merge_drugtweets.ipynb. After this step you should have all tweets prepared for the 4 drugs for later analyses.