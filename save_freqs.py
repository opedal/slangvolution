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
import sys

#andreas' token:
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAM0TSAEAAAAA%2BgyH%2F7NXwQnQ%2FyT0ebZ5nsQ3N5Y%3DtW4YxDF7ByzGMCpW0pvIPMFuSrpRq4mIXpPoEePyQSloe0WfZt" # INSERT TOKEN

#Daphna's bearer token
#BEARER_TOKEN =  "AAAAAAAAAAAAAAAAAAAAAL7hOgEAAAAAvM92PZSwVJ%2Ba%2BOD5Pgi4N298uTI%3DBBY7UCntIx9eXHBqGRjgjQcjoDFlMgFGJCjzd65uKISX8VFpwc"

def average_frequency():
    file_names = { "slang 2020" : 'data/frequencies/freq_slang_counts_24h_2020.csv',
                   "slang 2010" : 'data/frequencies/freq_slang_counts_24h_2010.csv',
                   "nonslang 2020": 'data/frequencies/freq_nonslang_counts_24h_2020.csv',
                   "nonslang 2010": 'data/frequencies/freq_nonslang_counts_24h_2010.csv',
    }
    avgs = {}
    for (k,fn) in file_names.items():
        df = pd.read_csv(fn)
        avg_freq = np.average(df.freq.values)
        print("average frequency for", k, "is", avg_freq)
        avgs[k] = avg_freq
    print("the frequency of nonslang words between 2010 and 2020, increased times ",
          avgs["nonslang 2020"]/avgs["nonslang 2010"])
    print("the frequency of slang words between 2010 and 2020, increased times ",
          avgs["slang 2020"]/avgs["slang 2010"])

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

def create_count_url(word,start_time,end_time, bucket="hour"):
    start_time = "start_time="+start_time #2021-01-17T00:00:00Z"
    end_time = "end_time="+end_time #2021-01-18T00:00:00Z"
    #max_results_str = "max_results="+str(max_results)
    query = word#"TwitterDev%20%5C%22search%20api%5C%22"
    granularity=""
    bucket = "bucket="+str(bucket)
    url = "https://api.twitter.com/2/tweets/counts/all?query={}&{}&{}".format(
        query, start_time, end_time
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

def approx_freq(word, year=2010,num_dates=20, hour_gap=6):
    FIRST_DATE = datetime.datetime(year,1,1)
    bearer_token = auth()
    headers = create_headers(bearer_token)
    date_spans = get_random_dates(start_date=FIRST_DATE, num_dates=num_dates,hour_gap=hour_gap)
    T, total_count = 0,0
    for dt_spn in date_spans:
        print("getting tweets for ", dt_spn[0])
        try:
            start_date = datetime2apidate(dt_spn[0])
            end_date = datetime2apidate(dt_spn[1])
            url = create_count_url(word, start_time=start_date, end_time=end_date)
            json_response, is_successful = connect_to_endpoint(url, headers)
            count = json_response["meta"]['total_tweet_count']
            if is_successful:
                T += 1
                total_count += count
        except:
            continue
    if T <= 0: return -1
    avg_count = total_count/T
    return avg_count

def check_example_words():
    #time.sleep(15*60)
    # for word in ["fam", "noob"]:
    #     # ["lowkey", "noob","chillax","bling", "bromance", ]
    #     for year in [2010,2020]:
    #         freq = approx_word_freq(word,
    #                                 year=year,
    #                                 num_dates=args.num_dates,
    #                                 hour_gap=2,
    #                                 )
    #         print("freq for",word, "in",year,"is", freq)
    return

if __name__ == '__main__':
    #words_of_interest = ["bromance", "bling","fam", "lowkey","unicorn", "they","performative","haircut","inclusive"]
    #["haircut", "inclusive", "bling", "chillax"]
    #["fam", "lowkey","unicorn", "they","performative","haircut", "inclusive", "bling", "chillax"]
    REQUEST_LIMIT = 300
    words_path = "word-lists/all_words_300.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="slang") #{"slang","nonslang","both"}
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--save-dir", type=str, default="data/frequencies/")
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--hour-gap",type=int,default=48)
    parser.add_argument("--num-dates",type=int,default=40)
    args = parser.parse_args()

    selected_words_df = pd.read_csv(words_path)
    words_list = list(selected_words_df[selected_words_df.type == args.type].word)
    PATHS = {"slang":"freq_slang_counts_24h_2020.csv",
             "nonslang":"freq_nonslang_counts_24h_2020.csv"
             }

    freq_file_path = os.path.join(args.save_dir, PATHS[args.type])
    #freq_file_path = "data/frequencies/words_of_interest_freqs.csv"
    print("saving word frequencies under", freq_file_path)
    num_words_until_pause = np.ceil(REQUEST_LIMIT/args.num_dates)
    hour_gap=24
    i = 0
    #time.sleep(15*60)
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
        i += 1
        print("saved tweets for", word)
        ## Update slang word dataframe so that we don't sample tweets from this word again
        #idx = np.where(selected_words_df.word == word)
        #selected_words_df.loc[idx[0][0], 'is_saved'] = True
        #selected_words_df.to_csv(words_path)
        ## Wait 15 minutes to avoid request rate restrictions
        if i == num_words_until_pause:
            i = 0
            print("will wait 15 minutes now")
            time.sleep(15*60)
            print("finished waiting")