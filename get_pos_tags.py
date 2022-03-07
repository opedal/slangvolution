import pandas as pd
import os.path as osp
# If you haven't yet, run these downloads:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from pos_tagging import count_tweets_per_word, pos_for_causal, analyse_pos_tags

if __name__ == '__main__':
    TWEET_PATH="data/"
    SAVE_PATH="data/POS/"
    causal_data_path = "data/causal_data.csv"

    # parameters for POS tagging
    PERCENT=True #use percent to determine the threshold
    COMBINE=True #combine 2010 & 2020 together
    MIN=5 #minimum for threshold

    tweets_per_word = count_tweets_per_word(tweet_path=TWEET_PATH)
    df_pos = pos_for_causal(tweets_per_word, save_path=SAVE_PATH, minimum=MIN, percent=PERCENT, combine=COMBINE)

    df_pos.to_csv(osp.join(SAVE_PATH, "pos_tags_min{}_{}.csv".format(MIN,("pc" if PERCENT else ""),
                                                            ("combined" if COMBINE else ""),
                                                            )))

    ## To merge the POS tags with the rest of the causal data
    causal_data = pd.read_csv("data/causal_data.csv")
    causal_df = causal_data[['word', 'freq2010', 'freq2020', 'type', 'semantic_change', 'polysemy']]

    df_causal = pd.merge(causal_df,
                            df_pos[["word", "most_common", "Noun_binary",
                                     "Verb_binary", "Adj_binary", "Adverb_binary"]],
                            on="word")

    df_causal.to_csv("data/causal_data_min{}_{}.csv".format(MIN,("pc" if PERCENT else ""),
                                                            ("combined" if COMBINE else ""),
                                                            ))

    analyse_pos_tags(save_path=SAVE_PATH, causal_data_path=causal_data_path)