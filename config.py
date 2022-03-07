
#-------------------- Visualizations --------------------#

SLANG_COLOR = "darkorange"
NONSLANG_COLOR = "mediumslateblue"
HYBRID_COLOR = "xkcd:bright sky blue"

#-------------------- Tweet & Representation Retrieval --------------------#

MIN_TWEETS_PER_WORD = 150
YEARLY_FREQ_NORMALIZATION_CONSTANTS = {2010: 1,
                           2011: 2,
                           2012: 5,
                           2013: 7,
                           2014: 6.8,
                           2015: 6.6,
                           2016: 6.4,
                           2017: 6.4,
                           2018: 6.2,
                           2019: 6.4,
                           2020: 6.4
                           }

#-------------------- Semantic Change --------------------#

SEMEVAL_PATH = ".../semeval2020_ulscd_eng/truth/graded.txt"
SEMEVAL_TARGETS_PATH = ".../semeval2020_ulscd_eng/targets.txt"

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

FREQ_FILE_NAMES = {"slang2020": 'freq_slang_counts_24h_2020.csv',
              "slang2010": 'freq_slang_counts_24h_2010.csv',
              "nonslang2020": 'freq_nonslang_counts_24h_2020.csv',
              "nonslang2010": 'freq_nonslang_counts_24h_2010.csv',
              "sample2020": 'freq_sample_words_24h_2020.csv',
              "sample2010": 'freq_sample_words_24h_2010.csv',
              "hybrid2010": "freq_hybrid_counts_24h_2010.csv",
              "hybrid2020": "freq_hybrid_counts_24h_2020.csv",
              }

POLYSEMY_FILE_NAMES = {"slang": "polysemy_slang.csv",
                       "nonslang": "polysemy_nonslang.csv",
                       "hybrid": "polysemy_hybrid.csv",
                       }

TWEET_FOLDER_NAMES = {'slang': {"old": "tweets_old/slang_word_tweets",
                              "new": "tweets_new/slang_word_tweets"
                              },
                    'nonslang': {
                               "old": "tweets_old/nonslang_word_tweets",
                               "new": "tweets_new/nonslang_word_tweets"
                           },
                    'hybrid': {
                          "old": "tweets_old/hybrid_word_tweets",
                          "new": "tweets_new/hybrid_word_tweets"
                      }
                    }