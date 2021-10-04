import requests
import os
import json
import datetime
import random
import pandas as pd
import numpy as np
# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
import time
import argparse
from os import path as osp
import sys

#andreas' token:
#BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAM0TSAEAAAAA%2BgyH%2F7NXwQnQ%2FyT0ebZ5nsQ3N5Y%3DtW4YxDF7ByzGMCpW0pvIPMFuSrpRq4mIXpPoEePyQSloe0WfZt" # INSERT TOKEN

#Daphna's bearer token
BEARER_TOKEN =  "AAAAAAAAAAAAAAAAAAAAAL7hOgEAAAAAvM92PZSwVJ%2Ba%2BOD5Pgi4N298uTI%3DBBY7UCntIx9eXHBqGRjgjQcjoDFlMgFGJCjzd65uKISX8VFpwc"

def random_sample_date(start_date,day_gap=365):
    td = random.random() * datetime.timedelta(days=day_gap)
    dt = start_date + td
    return dt

def get_random_dates(start_date, num_dates=50, hour_gap=24):
    date_spans = []
    for i in range(num_dates):
        date = random_sample_date(start_date=start_date)
        date_spans.append((date, date+ datetime.timedelta(hours=hour_gap)))
    return date_spans

def format_two_digits(num_str):
    if len(num_str) == 1:
        return '0'+num_str
    return num_str

def datetime2apidate(dt):
    year = str(dt.year)
    month = format_two_digits(str(dt.month))
    day = format_two_digits(str(dt.day))
    hour = format_two_digits(str(dt.hour))
    minute = format_two_digits(str(dt.minute))
    seconds = "00.000Z"
    date_day = "-".join([year,month,day])
    time_in_day = ":".join([hour,minute,seconds])
    return date_day + "T" + time_in_day

def auth():
    return BEARER_TOKEN

def create_url(word,start_time,end_time):
    query = word+", lang:en"
    #"from:twitterdev -is:retweet"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    start_time = "start_time="+start_time #2021-01-17T00:00:00Z"
    end_time = "end_time="+end_time #2021-01-18T00:00:00Z"
    tweet_fields = "tweet.fields=author_id,created_at"
    url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&{}&{}".format(
        query, start_time, end_time, tweet_fields
    )
    return url

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print("response status:", response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json(), (response.status_code == 200)

def get_year(utc_str):
    return utc_str[:4]

def get_month(utc_str):
    return utc_str[5:7]

def get_day(utc_str):
    return utc_str[8:10]

def update_tweet_df(json_response, tweets_df, word):
    data = json_response['data']
    for tweet in data:
        tweets_df['word'].append(word)
        tweets_df['id'].append(tweet['id'])
        tweets_df['year'].append(get_year(tweet['created_at']))
        tweets_df['month'].append(get_month(tweet['created_at']))
        tweets_df['day'].append(get_day(tweet['created_at']))
        tweets_df['text'].append(tweet['text'])
        tweets_df['author_id'].append(tweet['author_id'])
    return tweets_df

def get_tweet_count(json_response):
    data = json_response['data']
    return len(data)

def get_tweets_df(df_path):
    tweet_columns = ['word',
                     'id',
                     'year',
                     'month',
                     'day',
                     'text',
                     'author_id']
    try:
        tweets_df = pd.read_csv(df_path)
        tweets_df = tweets_df[tweet_columns]
        tweets_df = tweets_df.to_dict(orient='list')
    except FileNotFoundError:
        tweets_df = {'word': [],
                     'id': [],
                     'year': [],
                     'month': [],
                     'day': [],
                     'text': [],
                     'author_id': [],
                     }
    return tweets_df

def gap4word(word, default_gap=48):
    # Setting larger spans for infrequent words
    gaps_per_word = {'whadja':240,
                     "YooKay":240,
                     "hasbian":240,
                     "tardnation":240,
                     "colitas":72,
                     "brotox":120,
                     "Brotox":120,
                     "punanni":120,
                     }
    if word in gaps_per_word:
        return gaps_per_word[word]
    else:
        return default_gap

def get_word_tweets_df(word='yeet',
                       year=2010,
                       save_path="data",
                       num_dates=20,
                       MIN_TWEETS_PER_WORD=200,
                       hour_gap=24,
                       ):
    df_path = osp.join(save_path, "tweets_df_" + str(word) + ".csv")
    tweets_df = get_tweets_df(df_path=df_path)
    if len(tweets_df["word"]) >= MIN_TWEETS_PER_WORD:
        print("there are enough tweets for", word)
        return False
    FIRST_DATE = datetime.datetime(year,1,1)
    bearer_token = auth()
    headers = create_headers(bearer_token)
    date_spans = get_random_dates(start_date=FIRST_DATE, num_dates=num_dates,
                                  hour_gap=gap4word(word, default_gap=hour_gap))
    for dt_spn in date_spans:
        print("getting tweets for ", dt_spn[0])
        try:
            start_date = datetime2apidate(dt_spn[0])
            end_date = datetime2apidate(dt_spn[1])
            url = create_url(word=word, start_time=start_date, end_time=end_date)
            json_response, is_successful = connect_to_endpoint(url, headers)
            tweets_df = update_tweet_df(json_response=json_response, tweets_df=tweets_df, word=word)
            #print(json.dumps(json_response, indent=4, sort_keys=True))
        except:
            continue
    tweets_df = pd.DataFrame(tweets_df)
    tweets_df = tweets_df.drop_duplicates('text')
    print("saving tweets for", word, "at", df_path)
    tweets_df.to_csv(df_path)
    return True

def approx_word_freq(word, year=2010, num_dates=11, hour_gap=0.5):
    FIRST_DATE = datetime.datetime(year, 1, 1)
    bearer_token = auth()
    headers = create_headers(bearer_token)
    date_spans = get_random_dates(start_date=FIRST_DATE, num_dates=num_dates, hour_gap=hour_gap)
    total_num_tweets_with_word = 0
    T = 0
    for dt_spn in date_spans:
        print("getting tweets for ", dt_spn)
        try:
            start_date = datetime2apidate(dt_spn[0])
            end_date = datetime2apidate(dt_spn[1])
            url = create_url(word=word, start_time=start_date, end_time=end_date)
            json_response, is_successful = connect_to_endpoint(url, headers)
            num_tweets_with_word = get_tweet_count(json_response)
            T += 1
            total_num_tweets_with_word += num_tweets_with_word
        except:
            continue
    if T == 0 :
        return total_num_tweets_with_word
    avg_num_tweets_with_word = total_num_tweets_with_word/T
    return avg_num_tweets_with_word

def save_word_freqs(words_list):
    original_stdout = sys.stdout
    freqs = {}
    for word in words_list:
        with open('freqs.txt', 'w') as freqs_file:
            sys.stdout = freqs_file  # Change the standard output to the file we created.
            freq = approx_word_freq(word)
            freqs[word] = freq
            print("frequency of ", word, "is", freq)
    sys.stdout = original_stdout

if __name__ == "__main__":
    NUM_DATES=20
    HOUR_GAP=48

    words_path = "word-lists/all_words_300.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="slang") #{"slang","nonslang","both"}
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--save-dir", type=str, default="data/")
    parser.add_argument("--iter", type=int, default=5)
    args = parser.parse_args()

    selected_words_df = pd.read_csv(words_path)
    words_list = list(selected_words_df[selected_words_df.type == args.type].word)
    PATHS = {2010: "tweets_old",
             2020:"tweets_new",
             "slang":"slang_word_tweets",
             "nonslang":"nonslang_word_tweets"
             }

    save_dir = os.path.join(args.save_dir,PATHS[args.year], PATHS[args.type])
    print("saving tweets under", save_dir)

    for k in range(0,args.iter):
        i = 0
        print("----- ", k, "-----")
        for word in words_list:
            if word == "YooKay" or word == "tardnation":
                continue
            print("getting tweets for", word)
            got_tweets = get_word_tweets_df(word, year=args.year,
                                            save_path=save_dir,
                                            num_dates=NUM_DATES,
                                            hour_gap=HOUR_GAP,
                                            )
            if got_tweets: i += 1
            print("saved tweets for", word)
            ## Update slang word dataframe so that we don't sample tweets from this word again
            idx = np.where(selected_words_df.word == word)
            selected_words_df.loc[idx[0][0], 'is_saved'] = True
            selected_words_df.to_csv(words_path)
            ## Wait 15 minutes to avoid request rate restrictions
            if i == 3:
                i = 0
                print("will wait 15 minutes now")
                time.sleep(15*60)
                print("finished waiting")
