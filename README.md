# Exploring Time Trends and Public Opinions on COVID-19-related Therapeutics on Twitter
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2201.07281-b31b1b.svg)](https://arxiv.org/pdf/2206.14358)

This repo contains the official code and analyses results for the paper <i>[Using Twitter Data to Understand Public Perceptions of Approved versus Off-label Use for COVID-19-related Medications](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocac114/6625661)</i> accepted by JAMIA 2022. We release the following resources:
- the NLP pipeline for analyzing public perception in drug use in this repository
- our `drug-stance-bert` model in an off-the-shell fashion, available at [HuggingFace](https://huggingface.co/ningkko/drug-stance-bert). 

If you use our pipeline or models, please kindly cite our work with

```
@article{10.1093/jamia/ocac114,
    author = {Hua, Yining and Jiang, Hang and Lin, Shixu and Yang, Jie and Plasek, Joseph M and Bates, David W and Zhou, Li},
    title = "{Using Twitter Data to Understand Public Perceptions of Approved versus Off-label Use for COVID-19-related Medications}",
    journal = {Journal of the American Medical Informatics Association},
    year = {2022},
    month = {07},
    issn = {1527-974X},
    doi = {10.1093/jamia/ocac114},
    url = {https://doi.org/10.1093/jamia/ocac114},
    note = {ocac114},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocac114/44371833/ocac114.pdf},
}

```
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
This study was approved by the Mass General Brigham International Review Board. The NLP pipeline is a joint work between [Yining Hua](https://ningkko.wordpress.com/about-me/) (@ningkko) and [Hang Jiang](https://www.mit.edu/~hjian42/) (@hjian42). We thank our coauthors Shixu Lin, Jie Yang, Joseph Plasek, David Bates, and Li Zhou. We also thank [MIT Center for Constructive Communication (CCC)](https://www.ccc.mit.edu/) for funding the American politician dataset from [Ballotpedia](https://ballotpedia.org/). 
