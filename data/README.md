Since the usage requirement of Twitter data includes not sharing tweet content publicly, we provide only the IDs of the tweets we used in this study. 

The tweets provided in this folder have gone through 1) keyword matching and 2) removal of tweets from users with no location info. Tweets with more than one drug are not removed for replicating the trend analysis.


UPDATE: It seems that hcq.csv and remdesivir.csv don't work properly. If you encounter any problems first try applying .astype('int64') to the ID column. If it doesn’t work please email me at yhua.research@gmail.com, and cc makiasagawa@gmail.com for the original data.



