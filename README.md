# Exploring Time Trends and Public Opinions on COVID-19-related Therapeutics on Twitter

This repo contains code and analyses results for [this paper]()

In this repo, we release the pipeline for analyzing public perception in drug use. Following the pipeline one sholud be able to analyze any drug.

We also release our BERT stance model so it can be used in an off-the-shell fashion without training. Model available at [HuggingFace](https://huggingface.co/ningkko/drug-stance-bert). 

### 1. Preprocessing
We have all preprocssing steps summarized in [its own README](./preprocessing/README.md). Follow the instructions.

### 2. Time trend analysis
Time trend analysis is in the time_trend folder. Follow Processor.ipynb to generate data for plotting the trends, and then use the plot_trend.py file to plot.

### 3. Stance detection
To train the stance models, follow the trainer, to run inference on tweets, follow inference.ipynb.

### 4. Geoinference
Use processor.ipynb to process the data we got from stance detection, unify user locations and calculate statewide average stance. Use the state_map.Rmd to visualize the results.

### 5. Content analysis
Follow clustering.ipynb to cluster and visualize tweets. Use NER.ipynb to find NERs and visualize the results in word clouds.