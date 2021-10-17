"""
Save approximate word frequencies
- A word's frequency is approximated by the average number of times it is tweeted per day
"""
import argparse
import os
import numpy as np
import time

from twitter_api import *
import config

if __name__ == '__main__':
    REQUEST_LIMIT = 300

    words_path = "data/word-lists/all_words_300.csv"
    PATHS = config.FREQ_FILE_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="slang") #{"slang","nonslang","both","sample"}
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--save-dir", type=str, default="data/frequencies/")
    parser.add_argument("--num-dates",type=int,default=40)
    args = parser.parse_args()

    selected_words_df = pd.read_csv(words_path)
    words_list = list(selected_words_df[selected_words_df.type == args.type].word)

    freq_file_path = os.path.join(args.save_dir, PATHS[args.type + str(args.year)])
    print("saving word frequencies under", freq_file_path)
    num_words_until_pause = np.ceil(REQUEST_LIMIT/args.num_dates)
    hour_gap=24
    num_words_since_pause = 0
    for word in words_list:
        word = word.lower()
        freq_df = pd.read_csv(freq_file_path)
        if word in freq_df.word.values:
            continue
        print("getting frequency for", word)
        freq = approx_freq(word, year=args.year,
                                 num_dates=args.num_dates,
                                 hour_gap=hour_gap,)
        if freq == -1:
            continue
        with open(freq_file_path, "a") as freq_file:
                row = (",").join([str("%.2f" % freq), word, str(args.year), args.type])
                freq_file.write("\n" + row)
        num_words_since_pause += 1
        print("saved tweets for", word)
        ## Wait 15 minutes to avoid request rate restrictions
        if num_words_since_pause == num_words_until_pause:
            num_words_since_pause = 0
            print("will wait 15 minutes now")
            time.sleep(15*60)
            print("finished waiting")