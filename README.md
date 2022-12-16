# Slangvolution
Repo for the paper [**Slangvolution: A Causal Analysis of Semantic Change and Frequency Dynamics in Slang**](https://aclanthology.org/2022.acl-long.101/), accepted for publication as a main conference paper at ACL 2022. This README will walk you through the code and how to reproduce the results.

Start by installing all libraries:

`pip install -r requirements.txt`

## Citation

```
@inproceedings{keidar-etal-2022-slangvolution,
    title = "Slangvolution: {A} Causal Analysis of Semantic Change and Frequency Dynamics in Slang",
    author = "Keidar, Daphna  and
      Opedal, Andreas  and
      Jin, Zhijing  and
      Sachan, Mrinmaya",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.101",
    doi = "10.18653/v1/2022.acl-long.101",
    pages = "1422--1442",
    abstract = "Languages are continuously undergoing changes, and the mechanisms that underlie these changes are still a matter of debate. In this work, we approach language evolution through the lens of causality in order to model not only how various distributional factors associate with language change, but how they causally affect it. In particular, we study slang, which is an informal language that is typically restricted to a specific group or social setting. We analyze the semantic change and frequency shift of slang words and compare them to those of standard, nonslang words. With causal discovery and causal inference techniques, we measure the effect that word type (slang/nonslang) has on both semantic change and frequency shift, as well as its relationship to frequency, polysemy and part of speech. Our analysis provides some new insights in the study of language change, e.g., we show that slang words undergo less semantic change but tend to have larger frequency shifts over time.",
}
```

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
The causal analysis requires a .csv file that includes all variables. This file can be reproduced by following the above steps. Additionally, we provide the file `data/causal_data_input.csv` which consists of all variables used for our analysis apart from POS. The POS tags can be achieved as described above. The causal discovery algorithm is done in the R script `causal_graph_learning.R`. It follows three main steps. We first import and preprocess the data. This includes the categorizations of the polysemy variable as is discussed in the paper. We then plot density and qq-plots for our variables. Finally, we perform the causal analysis with PC-stable (for various alpha values), and visualize the resulting causal graph.

## Other

You will find global variables in `config.py`, helper functions in `utils.py` and visualization tools in `visualizations.py`.

