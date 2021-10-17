
#-------------------- Visualizations --------------------#
SLANG_COLOR = "darkorange"
NONSLANG_COLOR = "mediumslateblue"
HYBRID_COLOR = "xkcd:bright sky blue"

#-------------------- Tweet & Representation Retrieval --------------------#

MIN_TWEETS_PER_WORD = 150

#-------------------- Semantic Change --------------------#
SEMEVAL_PATH = "/Users/alacrity/Documents/uni/CSNLP/project/semeval2020_ulscd_eng/truth/graded.txt"
SEMEVAL_TARGETS_PATH = "/Users/alacrity/Documents/uni/CSNLP/project/semeval2020_ulscd_eng/targets.txt"

#-------------------- File Naming --------------------#

REPR_FILE_NAMES = {'slang': {"old": "old_slang_reps.pickle",
                        "new": "new_slang_reps.pickle"
                             },
                  'nonslang': {
                      "old": "old_nonslang_reps.pickle",
                      "new": "new_nonslang_reps.pickle"
                            },
                  'hybrid': {
                      "old": "old_hybrid_reps.pickle",
                      "new": "new_hybrid_reps.pickle"
                  }
                   }

TWEET_FILE_NAMES = {'slang': {"old": "old_slang_tweets.pickle",
                              "new": "new_slang_tweets.pickle"
                              },
                    'nonslang': {
                               "old": "old_nonslang_tweets.pickle",
                               "new": "new_nonslang_tweets.pickle"
                           }
                    }

FREQ_FILE_NAMES = {"slang 2020": 'freq_slang_counts_24h_2020.csv',
              "slang 2010": 'freq_slang_counts_24h_2010.csv',
              "nonslang 2020": 'freq_nonslang_counts_24h_2020.csv',
              "nonslang 2010": 'freq_nonslang_counts_24h_2010.csv',
              "sample 2020": 'freq_sample_words_24h_2020.csv',
              "sample 2010": 'freq_sample_words_24h_2010.csv',
              "hybrid 2010": "freq_hybrid_counts_24h_2010.csv",
              "hybrid 2020": "freq_hybrid_counts_24h_2020.csv",
              }

