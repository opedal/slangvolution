import argparse
import os
import numpy as np
import time

from twitter_api import *

if __name__ == "__main__":

    # request limit of the api
    REQUEST_LIMIT=50

    PATHS = {2010: "tweets_old",
             2020:"tweets_new",
             "slang":"slang_word_tweets",
             "nonslang":"nonslang_word_tweets",
             "both":"hybrid_word_tweets"
             }

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="slang") #{"slang","nonslang","both"}
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--save-dir", type=str, default="data/")
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--hour-gap",type=int,default=48)
    parser.add_argument("--num-dates",type=int,default=20)
    parser.add_argument("--max-results",type=int,default=30)
    parser.add_argument("--words", type=str, default="all_words.csv")
    args = parser.parse_args()

    words_list = get_words_list(word_type=args.type, words_path=args.words)
    save_dir = os.path.join(args.save_dir, PATHS[args.year], PATHS[args.type])
    print("saving tweets under", save_dir)

    num_words_until_pause = np.ceil(REQUEST_LIMIT/args.num_dates) + 1

    for k in range(0,args.iter):
        num_words_since_pause = 0
        print("----- ", k, "-----")
        for word in words_list:
            print("getting tweets for", word)
            got_tweets = get_word_tweets_df(word,
                                            year=args.year,
                                            save_path=save_dir,
                                            num_dates=args.num_dates,
                                            hour_gap=args.hour_gap,
                                            max_results_per_response=args.max_results
                                            )
            if got_tweets: num_words_since_pause += 1
            print("saved tweets for", word)
            ## Wait 15 minutes to avoid request rate restrictions
            if num_words_since_pause == num_words_until_pause:
                print("will wait 15 minutes now")
                time.sleep(15*60)
                print("finished waiting")
                num_words_since_pause = 0
