# Exploring Time Trends and Public Opinions on COVID-19-related Therapeutics on Twitter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2201.07281-b31b1b.svg)](https://arxiv.org/pdf/2206.14358)

This repo contains code and analyses results for the paper <i>Using Twitter Data to Understand Public Perceptions of Approved versus Off-label Use for COVID-19-related Medications</i> accepted by JAMIA 2022. We include the following resources:
- the NLP pipeline for analyzing public perception in drug use in this repository
- our `drug-stance-bert` model in an off-the-shell fashion, available at [HuggingFace](https://huggingface.co/ningkko/drug-stance-bert). 

## NLP Pipeline

### 1. Preprocessing
We have all preprocssing steps summarized in [its own README](./preprocessing/README.md). Follow the instructions.

### 2. Time trend analysis
Time trend analysis is in the time_trend folder. Follow `Processor.ipynb` to generate data for plotting the trends, and then use the `plot_trend.py` file to plot.

### 3. Stance detection
To train the stance models, follow the trainer, to run inference on tweets, follow `inference.ipynb`.

### 4. Geoinference
Use `processor.ipynb` to process the data we got from stance detection, unify user locations and calculate statewide average stance. Use the `state_map.Rmd` to visualize the results.

### 5. Content analysis
Follow `clustering.ipynb` to cluster and visualize tweets. Use `NER.ipynb` to find NERs and visualize the results in word clouds.

### 6. Demographic analysis
Follow the README.md file inside `demographic_analysis` to replicate the demographic and political orientation inference and analysis.

## Acknowledgement
The NLP pipeline is a joint work between [Yining Hua](https://ningkko.wordpress.com/about-me/) (@ningkko) and [Hang Jiang](https://www.mit.edu/~hjian42/) (@hjian42). For questions regarding to steps 1, 2, 3, and 5, please contact [Yining Hua](https://ningkko.wordpress.com/about-me/) (@ningkko). For questions regarding to steps 4 and 6, please contact [Hang Jiang](https://www.mit.edu/~hjian42/) (@hjian42). We want to thank [MIT Center for Constructive Communication (CCC)](https://www.ccc.mit.edu/) for funding the American politician dataset from [Ballotpedia](https://ballotpedia.org/). 
