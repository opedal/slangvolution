"""
Code for accessing the Twitter API to retrieve tweets & frequency approximations
"""
import requests
import datetime
import random
import pandas as pd
from os import path as osp

#-------------------- Authentication & Connection to the API --------------------#

BEARER_TOKEN = ""

def auth():
    return BEARER_TOKEN

def create_url(word,start_time, end_time, max_results=500):
    """
    Create url for retrieving tweets
    """
    query = word+", lang:en"
    # Tweet fields are adjustable.
    # Options include: attachments, author_id, context_annotations, conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics, possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    start_time = "start_time="+start_time #2021-01-17T00:00:00Z"
    end_time = "end_time="+end_time #2021-01-18T00:00:00Z"
    max_results_str = "max_results="+str(max_results)
    tweet_fields = "tweet.fields=created_at"
    url = "https://api.twitter.com/2/tweets/search/all?query={}&{}&{}&{}&{}".format(
        query, start_time, end_time, max_results_str, tweet_fields
    )
    return url

def create_count_url(word,start_time,end_time, bucket="hour"):
    """
    Create url for retrieving the number of tweets containting a word
    """
    start_time = "start_time="+start_time #format 2021-01-17T00:00:00Z"
    end_time = "end_time="+end_time #format 2021-01-18T00:00:00Z"
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

#-------------------- Date Sampling & Formatting --------------------#

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

def get_year(utc_str):
    return utc_str[:4]

def get_month(utc_str):
    return utc_str[5:7]

def get_day(utc_str):
    return utc_str[8:10]

#-------------------- Updating the Tweet DataFrame --------------------#

def update_tweet_df(json_response, tweets_df, word):
    data = json_response['data']
    for tweet in data:
        tweets_df['word'].append(word)
        tweets_df['id'].append(tweet['id'])
        tweets_df['year'].append(get_year(tweet['created_at']))
        tweets_df['month'].append(get_month(tweet['created_at']))
        tweets_df['day'].append(get_day(tweet['created_at']))
        tweets_df['text'].append(tweet['text'])
    return tweets_df

def get_tweet_count(json_response):
    data = json_response['data']
    return len(data)

def get_tweets_df(df_path):
    tweet_columns = ['word',
                     #'id',
                     'year',
                     'month',
                     'day',
                     'text',
                     ]
    try:
        tweets_df = pd.read_csv(df_path)
        tweets_df = tweets_df[tweet_columns]
        tweets_df = tweets_df.to_dict(orient='list')
    except FileNotFoundError:
        tweets_df = {'word': [],
                     #'id': [],
                     'year': [],
                     'month': [],
                     'day': [],
                     'text': [],
                     }
    return tweets_df

def get_word_tweets_df(word='yeet',
                       year=2010,
                       save_path="data",
                       num_dates=20,
                       MIN_TWEETS_PER_WORD=200,
                       hour_gap=24,
                       max_results_per_response=500,
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
                                  hour_gap=hour_gap)
    for dt_spn in date_spans:
        try:
            start_date = datetime2apidate(dt_spn[0])
            end_date = datetime2apidate(dt_spn[1])
            url = create_url(word=word, start_time=start_date, end_time=end_date,max_results=max_results_per_response)
            json_response, is_successful = connect_to_endpoint(url, headers)
            tweets_df = update_tweet_df(json_response=json_response, tweets_df=tweets_df, word=word)
        except:
            continue
    tweets_df = pd.DataFrame(tweets_df)
    tweets_df = tweets_df.drop_duplicates('text')
    print("saving tweets for", word, "at", df_path)
    tweets_df.to_csv(df_path)
    return True

def approx_freq(word, year=2010, num_dates=40, hour_gap=6):
    FIRST_DATE = datetime.datetime(year,1,1)
    bearer_token = auth()
    headers = create_headers(bearer_token)
    date_spans = get_random_dates(start_date=FIRST_DATE, num_dates=num_dates, hour_gap=hour_gap)
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

def get_words_list(word_type, words_path="data/word-lists/all_words_300.csv"):
    selected_words_df = pd.read_csv(words_path)
    words_list = list(selected_words_df[selected_words_df.type == word_type].word)
    return words_list
