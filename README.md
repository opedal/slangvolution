# Slangvolution
Git repo for the paper "Slangvolution: A Causal Analysis of Semantic Change and Frequency Dynamics in Slang"

## Data Preparation

All the words used in this study are listed under `data/word-lists/all_words_300.csv`

To retrieve tweets run: 

`get_tweets.py --type slang --year 2020 --num-dates 40 --hour-gap 24 --max-results 50`

And for approximating word frequency run: 

`get_word_freqs.py --type slang --year 2020 --num-dates 40`

What each parameter means: 
- type: string corresponding to word type, can be slang, nonslang, or both
- year: 2020 or 2010
- num-dates: integer, number of random time points to sample per year 
- hour-gap: integer, number of hours to consider for tweet collection, after each random time point 
- max-results: integer, maximum number of tweets to get per request 

To preprocess the Urban Dictionary (UD) data run: 

`UD_data_preprocessing.py --path path-to-orig-UD-definitions`

(note that UD definitions need to first be downloaded)

## Representation Retrieval

First fine tune RoBERTa on the UD data: `MLM_fine_tuning.py --num-epochs 10 --patience 3`

Then, apply RoBERTa to the tweets to get the representations: `representations_retrieval.py --type slang`

## Semantic Change and Frequency Shift Scores

To get the APD scores on representations reduced to 100 dimensions with PCA run:

`semantic_change_scores.py --method apd --reduction pca`

Once the frequencies are saved, run 

`frequency_change_analysis.py`

To save dataframes with the frequency change statistics for each word type 
