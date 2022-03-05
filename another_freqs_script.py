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
from matplotlib import pyplot as plt

from nltk.corpus import words

#andreas' token:
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAM0TSAEAAAAA%2BgyH%2F7NXwQnQ%2FyT0ebZ5nsQ3N5Y%3DtW4YxDF7ByzGMCpW0pvIPMFuSrpRq4mIXpPoEePyQSloe0WfZt" # INSERT TOKEN

#Daphna's bearer token
#BEARER_TOKEN =  "AAAAAAAAAAAAAAAAAAAAAL7hOgEAAAAAvM92PZSwVJ%2Ba%2BOD5Pgi4N298uTI%3DBBY7UCntIx9eXHBqGRjgjQcjoDFlMgFGJCjzd65uKISX8VFpwc"

NORMALIZATION_CONSTANTS = {2010: 1,
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

def plot_yearly_freq_both(freq_df):
    from visualizations import SLANG_COLOR, NONSLANG_COLOR
    word = "duckface"
    freq_norm_word = freq_df.freq_norm[freq_df.word == word]
    plt.plot(freq_df.year[freq_df.word == word], freq_norm_word/max(freq_norm_word),
             label=word,
             color=SLANG_COLOR)

    # celeb = "celebutante"
    # freq_norm_celeb = freq_df.freq_norm[freq_df.word == celeb]
    # plt.plot(freq_df.year[freq_df.word == celeb], freq_norm_celeb/ max(freq_norm_celeb), label=celeb,
    #          color="xkcd:coral")

    # word2 = "incel"
    # freq_norm_word2 = freq_df.freq_norm[freq_df.word == word2]
    # plt.plot(freq_df.year[freq_df.word == word2], freq_norm_word2 / max(freq_norm_word2), label=word2,
    #          color="xkcd:auburn")

    word1 = "inclusive"
    freq_norm_word1 = freq_df.freq_norm[freq_df.word == word1]
    plt.plot(freq_df.year[freq_df.word == word1], freq_norm_word1/max(freq_norm_word1),
             label=word1,
             color=NONSLANG_COLOR)

    # unicorn = "unicorn"
    # freq_norm_unicorn = freq_df.freq_norm[freq_df.word == unicorn]
    # plt.plot(freq_df.year[freq_df.word == unicorn], freq_norm_unicorn / max(freq_norm_unicorn), label=unicorn,
    #          color="xkcd:azure")

    # word3 = "lawlessness"
    # freq_norm_word3 = freq_df.freq_norm[freq_df.word == word3]
    # plt.plot(freq_df.year[freq_df.word == word3], freq_norm_word3 / max(freq_norm_word3), label=word3,
    #          color="xkcd:electric blue")

    # didot = "didot"
    # freq_norm_didot = freq_df.freq_norm[freq_df.word == didot]
    # plt.plot(freq_df.year[freq_df.word == didot], freq_norm_didot / max(freq_norm_didot), label=didot,
    #          color="xkcd:robin's egg")
    #
    # anticlock = "anticlockwise"
    # freq_norm_anticlock = freq_df.freq_norm[freq_df.word == anticlock]
    # plt.plot(freq_df.year[freq_df.word == anticlock], freq_norm_anticlock / max(freq_norm_anticlock), label=anticlock,
    #          color="xkcd:light navy")

    plt.legend()
    #plt.title("Yearly relative frequency of the fastest increasing slang and nonslang words")
    plt.title("Yearly relative frequency of \"{}\",\"{}\"".format(word, word1))
    plt.savefig("duckface_inclusive_highres.png", dpi=300)
    #plt.show()

def plot_yearly_freq(freq_df, word="inclusive",word1=None):
    if word1 is None:
        plt.plot(freq_df.year[freq_df.word == word], freq_df.freq_norm[freq_df.word == word])
        plt.title("yearly frequency of \"{}\"".format(word))
        plt.show()
    else:
        word = "duckface"
        plt.plot(freq_df.year[freq_df.word == word], freq_df.freq_norm[freq_df.word == word], label=word)
        word1 = "inclusive"
        plt.plot(freq_df.year[freq_df.word == word1], freq_df.freq_norm[freq_df.word == word1], label=word1)
        plt.legend()
        plt.title("yearly frequency of \"{}\" and \"{}\"".format(word, word1))
        plt.show()

    # word = "duckface"
    # plt.plot(freq_df.year[freq_df.word == word], (freq_df.freq_norm[freq_df.word == word]) / max_duckface, label=word,
    #          color="darkorange")
    # word1 = "inclusive"
    # plt.plot(freq_df.year[freq_df.word == word1], (freq_df.freq_norm[freq_df.word == word1]) / max_inclusive,
    #          label=word1, color="mediumslateblue")
    # plt.legend()
    # plt.ylabel("relative frequency")
    # plt.xlabel("year")
    # plt.title("Yearly frequency of \"{}\" and \"{}\"".format(word, word1), fontweight="bold")
    # plt.show()

def plot_all_yearly_freqs():

    didot = "gotsta"
    freq_norm_didot = freq_df.freq_norm[freq_df.word == didot]
    plt.plot(freq_df.year[freq_df.word == didot], freq_norm_didot / max(freq_norm_didot), label=didot,
             color="gold")

    word1 = "meme"
    freq_norm_word1 = freq_df.freq_norm[freq_df.word == word1]
    plt.plot(freq_df.year[freq_df.word == word1], freq_norm_word1 / max(freq_norm_word1), label=word1, color="orange")

    word2 = "incel"
    freq_norm_word2 = freq_df.freq_norm[freq_df.word == word2]
    plt.plot(freq_df.year[freq_df.word == word2], freq_norm_word2 / max(freq_norm_word2), label=word2,
             color="xkcd:dark orange")

    word3 = "lawlessness"
    freq_norm_word3 = freq_df.freq_norm[freq_df.word == word3]
    plt.plot(freq_df.year[freq_df.word == word3], freq_norm_word3 / max(freq_norm_word3), label=word3,
             color="xkcd:electric blue")

    unicorn = "unicorn"
    freq_norm_unicorn = freq_df.freq_norm[freq_df.word == unicorn]
    plt.plot(freq_df.year[freq_df.word == unicorn], freq_norm_unicorn / max(freq_norm_unicorn), label=unicorn,
             color="xkcd:azure")

    anticlock = "anticlockwise"
    freq_norm_anticlock = freq_df.freq_norm[freq_df.word == anticlock]
    plt.plot(freq_df.year[freq_df.word == anticlock], freq_norm_anticlock / max(freq_norm_anticlock), label=anticlock,
             color="xkcd:light navy")

    plt.legend()
    plt.title("Yearly relative frequency of the fastest increasing slang and nonslang words")
    plt.show()

if __name__ == '__main__':
    REQUEST_LIMIT = 300

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="example") #{"slang","nonslang","both"}
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--save-dir", type=str, default="data/frequencies/")
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--hour-gap",type=int,default=48)
    parser.add_argument("--num-dates",type=int,default=40)
    args = parser.parse_args()

    freq_file_path = os.path.join(args.save_dir, "example_words.csv")
    freq_df = pd.read_csv(freq_file_path)

    freq_df["freq_norm"] = freq_df[["freq","year"]].apply(
        lambda x : x["freq"]/NORMALIZATION_CONSTANTS[x["year"]],
        axis=1)
    plot_yearly_freq_both(freq_df)
    plot_yearly_freq(freq_df, word="inclusive", word1="duckface")
    print("saving word frequencies under", freq_file_path)
    num_words_until_pause = np.ceil(REQUEST_LIMIT/args.num_dates)
    hour_gap=24
    i = 0
    #"mistreatment"
    for word in ["sculptured", "dysthymic","telogen"]:#["prettyful", "slore", "gotsta", "dafuq", "meme"]:#"anticlockwise", "didot", "lawlessness","celebutante"]:
        for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]:
            word = word.lower()
            freq_df = pd.read_csv(freq_file_path)
            print("getting frequency for", word)
            freq = approx_freq(word, year=year,
                                     num_dates=args.num_dates,
                                     hour_gap=hour_gap,)
            if freq == -1:
                continue
            with open(freq_file_path, "a") as freq_file:
                    row = (",").join([str("%.2f" % freq), word, str(year), args.type])
                    freq_file.write("\n" + row)
            i += 1
            print("saved tweets for", word)
            if i == num_words_until_pause:
                i = 0
                print("will wait 15 minutes now")
                time.sleep(15*60)
                print("finished waiting")