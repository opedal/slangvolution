"""
Script for filtering the Urban Dictionary data to get 100000 definitions out of the original 3534966
Filtering according to
- upvote/downvote ratio
- number of upovotes
And finally taking a random sample
"""
import pandas as pd
import time
import argparse
import os
SEED=101

def defs_to_pandas(fpath):
    with open(fpath) as f:
        defs = [line.split(sep='|') for line in f]

    num_defs = len(defs[1:])

    words, timestamps, authors, meanings = [""]*num_defs, [0]*num_defs, [""]*num_defs, [""]*num_defs
    examples, dates = [""]*num_defs, [""]*num_defs
    numLikes, numDislikes, tagLists = [0]*num_defs, [0]*num_defs, [""]*num_defs

    for i, line in enumerate(defs[1:]):
        words[i], timestamps[i], authors[i], meanings[i], \
        examples[i], dates[i], numLikes[i], numDislikes[i], tagLists[i] = line

    defs_df = pd.DataFrame({
        "word":words,
        "timestamp":timestamps,
        "author":authors,
        "meaning":meanings,
        "example":examples,
        "date":dates,
        "num_likes":numLikes,
        "num_dislikes":numDislikes,
        "taglist":tagLists
    })

    defs_df["date"] = defs_df.date.apply(process_date)
    defs_df["word"] = defs_df.word.apply(lambda x : x.replace('%20',' '))
    defs_df["num_likes"] = defs_df.num_likes.apply(str_to_int)
    defs_df["num_dislikes"] = defs_df.num_dislikes.apply(str_to_int)
    #defs_df["timestamp_date"] = defs_df.timestamp.apply(timestamp_to_date)
    defs_df.to_csv("data/defs.csv")
    return defs_df

def process_date(date_string):
    """
    turn date of format day-month-year into format year-month-day
    e.g. '05-03-2019' turns into '2019-03-05'
    """
    day = date_string[:2]
    month = date_string[3:5]
    year = date_string[6:]
    return ("-").join([year,month,day])

def str_to_int(x):
    try:
        return int(x)
    except:
        return -1

def timestamp_to_date(x):
    try:
        return time.ctime(int(str(x)[:-3]))
    except:
        return -1

def filter_sample_and_write(df, min_likes, min_ratio, name, sample_size=10000):
    """
        filters by minimum number of likes & min ratio of likes to dislikes
        and exports a subsample to csv
    """
    df["like_ratio"] = df["num_likes"] / df["num_dislikes"]
    newdf = df[(df["num_likes"] >= min_likes) & (df["like_ratio"] >= min_ratio)]
    newdf = newdf.sample(n=sample_size, random_state=SEED)
    print(df.shape, newdf.shape)
    if not os.path.exists("data"):
        os.mkdir("data")
    newdf.to_csv(f"data/{name}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/all_definitions.dat")
    parser.add_argument("--name", type=str, default="UD_filtered_100000_sampled")
    parser.add_argument("--min-likes", type=int, default=20)
    parser.add_argument("--min-ratio", type=int, default=2)
    parser.add_argument("--sample", type=int, default=100000)
    args = parser.parse_args()
    fpath = args.path
    defs = defs_to_pandas(fpath)
    filter_sample_and_write(defs, args.min_likes, args.min_ratio, name=args.name, sample_size=args.sample)