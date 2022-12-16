# Slangvolution
Repo for the paper [**Slangvolution: A Causal Analysis of Semantic Change and Frequency Dynamics in Slang**](https://aclanthology.org/2022.acl-long.101/), accepted for publication as a main conference paper at ACL 2022. This README will walk you through the code and how to reproduce the results.


Start by installing all libraries:

`pip install -r requirements.txt`

## Data Preparation

All the words used in this study are listed under `data/all_words.csv`. You may access tweets, frequency approximations, models and representations [here](https://polybox.ethz.ch/index.php/s/WOIZTYRzhPjho9j).

If you want to retrieve tweets, add your bearer token to `twitter_api.py` (you will need access to Twitter's API). Then run:

`get_tweets.py --type slang --year 2020 --num-dates 40 --hour-gap 24 --max-results 50`

To retrieve word frequencies, run: 

`get_word_frequencies.py --type slang --year 2020 --num-dates 40`

What each parameter means: 
- type: string corresponding to word type, can be slang, nonslang, or both
- year: 2020 or 2010
- num-dates: integer, number of random time points to sample per year 
- hour-gap: integer, number of hours to consider for tweet collection, after each random time point 
- max-results: integer, maximum number of tweets to get per request 

The Urban Dictionary data used for fine-tuning is taken from [**Urban Dictionary Embeddings for Slang NLP Applications**](https://aclanthology.org/2020.lrec-1.586/). Please reach out to the original authors for access. Then run:

`python UD_data_preprocessing.py --path PATH-TO-DATA/all_definitions.dat`

This will provide you with a filtered UD csv file, used for fine-tuning.

## Representation Retrieval

With the data from the previous step, run `python MLM_fine_tuning.py` to retrieve four models (with different learning rates). We recommend doing this remotely on a GPU (it takes a couple of days), but if you just wanna make sure that the code runs &mdash; add `--small True` and set the number of epochs to be small `--num-epochs 1`.

Then, apply RoBERTa to the tweets to get the representations: `representations_retrieval.py --type slang`. By default this will give you the summed representations across layers. Provide the data path to the new/old slang/nonslang/hybrid tweets with `--data-path data/tweets_new/slang_word_tweets`. The script will write two files, one for the representations and one for the tweet texts. Feel free to reach out to us for access to these files. 

You may also get the representations for the SemEval 2020 Task 1 data with the same script, by adding `--sem-eval True`. Download the data from [here](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/).

## POS Tags 

To get the POS tags from the tweets, run `python get_pos_tags.py`.

## Semantic Change and Frequency Shift Scores

To get the APD scores on representations reduced to 100 dimensions with PCA run:

`get_semantic_change_scores.py --method apd --reduction pca --type slang`

If you would like to experiment with different semantic change score metrics, you will find the relevant code in `semantic_change.py`.

The code for the frequency shift scores is provided in `frequency_change_analysis.py`. 

## Causal Analysis
The causal analysis requires a .csv file that includes all variables. This file can be reproduced by following the above steps. Additionally, we provide the file `data/causal_dataset.csv` which consists of all variables used for our analysis apart from POS. The POS tags can be achieved as described above. The causal discovery algorithm is done in the R script `causal_graph_learning.R`. It follows three main steps. We first import and preprocess the data. This includes the categorizations of the polysemy variable as is discussed in the paper. We then plot density and qq-plots for our variables. Finally, we perform the causal analysis with PC-stable (for various alpha values), and visualize the resulting causal graph.

## Other

You will find global variables in `config.py`, helper functions in `utils.py` and visualization tools in `visualizations.py`.

