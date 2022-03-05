import nltk
from nltk.tokenize import word_tokenize
import os
from os import path as osp
import pandas as pd
from config import TWEET_FOLDER_NAMES
from collections import Counter
# If you haven't yet, run these downloads:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
TWEET_PATH = "/Users/alacrity/Documents/Side Projects/Slangvolution/data/tweets/"
SAVE_PATH = "data/POS/"

unify_pos = {"NN":"Noun", "NNS":"Noun", "NNP":"Noun",
             "VB":"Verb", "VBD":"Verb", "VBP":"Verb", "VBN":"Verb", "VBG":"Verb", "VBZ":"Verb",
             "RB":"Adverb", "RBR":"Adverb", "RBS":"Adverb",
             "RP":"Particle",
             "PRP$":"Pronoun","PRP":"Pronoun",
             "JJ":"Adj", "JJR":"Adj", "JJS":"Adj",
             "SYM":"Symbol",
             "CC":"Conjunction","IN":"Conjunction",
             "CD":"Num",
             "DT":"Det","PDT":"Pre-determiner",
             "FW":"Foreign",
             "":"na","''":"na",
             }

def get_unified_pos(pos):
    if pos in unify_pos:
        return unify_pos[pos]
    return pos

def get_pos(sentence,word=""):
    if type(sentence) != str:
        return "na"
    sentence = sentence.lower()
    text = word_tokenize(sentence)
    POS = nltk.pos_tag(text)
    POS_w = [pos for (wrd, pos) in POS if wrd == word]
    if len(POS_w) == 0:
        return "na"
    return POS_w[0]

def update_top_pos_df(top_pos_df, word_pos_counter, curr_word, save_path):
    most_common_pos = max(word_pos_counter, key=lambda x: word_pos_counter[x])
    top_pos_df["word"].append(curr_word)
    top_pos_df["most_common_pos"].append(most_common_pos)
    curr_pos_df = pd.DataFrame(top_pos_df)
    curr_pos_df.to_csv(osp.join(save_path, "most_common_pos.csv"))
    return top_pos_df

def update_pos_df(pos_df, word_pos_counter, curr_word, save_path):
    pos_df["word"].append(curr_word)
    for k in pos_df.keys():
        if k != "word":
            pos_df[k].append(word_pos_counter[k])
    curr_pos_df = pd.DataFrame(pos_df)
    curr_pos_df.to_csv(osp.join(save_path, "all_pos.csv"))
    return pos_df

def save_pos_tags():
    for type in ["slang","nonslang"]:
        for time in ["old","new"]:
            top_pos_df = {"word": [],
                          "most_common_pos": [],
                          }
            pos_df = {"word": [], "Noun": [], "Verb": [], "Adverb": [], "Adj": [], "Det": [],
                      "Particle": [], "Num": [], "Symbol": [], "Foreign": [], "Conjunction": [],
                      }
            save_path = osp.join(SAVE_PATH, time, type)
            tweets_folder = osp.join(TWEET_PATH, TWEET_FOLDER_NAMES[type][time])
            print("getting tweets from", tweets_folder)
            for tweets_file in sorted(os.listdir(tweets_folder)):
                tweets = pd.read_csv(osp.join(tweets_folder, tweets_file))
                if len(tweets) == 0:
                    continue
                curr_word = tweets.word[0].lower()
                # try:
                tweets["POS"] = tweets["text"].apply(lambda x : get_pos(x,word=curr_word))
                tweets["POS_unified"] = tweets["POS"].apply(get_unified_pos)
                word_pos_counter = Counter(tweets.POS_unified)
                top_pos_df = update_top_pos_df(top_pos_df=top_pos_df,curr_word=curr_word,word_pos_counter=word_pos_counter,save_path=save_path)
                pos_df = update_pos_df(pos_df=pos_df,curr_word=curr_word,word_pos_counter=word_pos_counter, save_path=save_path)
                # except:
                #     print("couldn't get tweet pos for",curr_word)

def count_tweets_per_word():
    tweets_per_word = {}
    for time in ["old", "new"]:
        tweets_per_word[time] = {}
        for type in ["slang","nonslang"]:
            tweets_folder = osp.join(TWEET_PATH, TWEET_FOLDER_NAMES[type][time])
            print("getting tweets from", tweets_folder)
            for tweets_file in sorted(os.listdir(tweets_folder)):
                tweets = pd.read_csv(osp.join(tweets_folder, tweets_file))
                if len(tweets) == 0:
                    continue
                curr_word = tweets.word[0].lower()
                num_tweets = len(tweets["text"])
                tweets_per_word[time][curr_word] = num_tweets
    return tweets_per_word

def analyse_pos_tags():
    causal_data = pd.read_csv("data/causal_data_input.csv")
    pos_accross_types = { }
    #"slang_old" :{},"slang_new":{}, "nonslang_old":{}, "nonslang_new": {},

    for type in ["slang", "nonslang"]:
        words = causal_data.word[causal_data.type == type]
        print(len(words),type,"words")
        for time in ["old", "new"]:
            df_folder_path = osp.join(SAVE_PATH, time, type)
            df_file_path = osp.join(df_folder_path, "most_common_pos.csv")
            df = pd.read_csv(df_file_path)
            most_common = Counter(df.most_common_pos[df.word.isin(words)])
            pos_accross_types[type + "_" + time] = most_common
    print(pos_accross_types)

def sum_pos(x, min=10):
    """
    Sum the number of times a pos tag was given in 2010, encoded by x[0]
     with the number of times it was given in 2020, encoded by x[1]
    Only consider the tag if it appeared more than min times in both periods
    If it hadn't appeared at least min times in either period, return 0
    """
    pos_tag_sum = x[0] + x[1]
    has_appeared_min_times = ((x[0]>min) and (x[1]>min))
    return pos_tag_sum*int(has_appeared_min_times)

def prep_for_causal(tweets_per_word, combine=True, minimum=10, percent=False):
    pos_tags = {"old": {}, "new": {}}
    for word_type in ["slang", "nonslang"]:
        for time in ["old", "new"]:
            df_folder_path = osp.join(SAVE_PATH, time, word_type)
            df_file_path = osp.join(df_folder_path, "all_pos.csv")
            df = pd.read_csv(df_file_path)
            pos_tags[time][word_type] = df

    df_old_slang = pos_tags["old"]["slang"]
    df_old_nonslang = pos_tags["old"]["nonslang"]
    df_new_slang = pos_tags["new"]["slang"]
    df_new_nonslang = pos_tags["new"]["nonslang"]

    df_slang = pd.merge(df_new_slang, df_old_slang, on="word")

    POS = ["Noun", "Verb", "Adverb", "Adj"]
    for pos in POS:
        if combine: df_slang[pos] = df_slang[pos + "_x"] + df_slang[pos + "_y"]
        else: df_slang[pos] = df_slang[[pos + "_x", pos + "_y"]].apply(lambda x : sum_pos(x, minimum),
                                                                       axis=1)
    df_slang['most_common'] = df_slang[POS].idxmax(axis=1)
    df_nonslang = pd.merge(df_new_nonslang, df_old_nonslang, on="word")

    for pos in POS:
        if combine: df_nonslang[pos] = df_nonslang[pos + "_x"] + df_nonslang[pos + "_y"]
        else: df_nonslang[pos] = df_nonslang[[pos + "_x", pos + "_y"]].apply(lambda x : sum_pos(x, minimum),
                                                                             axis=1)
    df_nonslang['most_common'] = df_nonslang[POS].idxmax(axis=1)

    df_all = pd.concat([df_slang, df_nonslang])
    df_all["num_tweets"] = df_all["word"].apply(
                            lambda wrd: tweets_per_word["old"][wrd] + tweets_per_word["new"][wrd])
    if combine: MIN = minimum
    else: MIN = 0
    for pos in POS:
        if percent:
            pc = MIN/100
            df_all[pos+"_binary"] = df_all[[pos,"num_tweets"]].apply(
                                            lambda x: int(x[0] > pc*x[1]), axis=1)
        else: df_all[pos + "_binary"] = df_all[pos].apply(lambda x: x > MIN)
    return df_all

if __name__ == '__main__':
    COMBINED = True
    MIN = 5
    PERCENT = True
    to_str = {True:"combined", False: "sep"}
    pc_to_str = {False : "", True : "percent"}

    tweets_per_word = count_tweets_per_word()
    df_all = prep_for_causal(tweets_per_word, combine=COMBINED, minimum=MIN, percent=PERCENT)
    causal_MW = pd.read_csv("data/causal_data_MW.csv")
    causal_df = causal_MW[['word', 'freq2010', 'freq2020', 'type', 'semantic_change', 'polysemy']]
    df_causal = pd.merge(causal_df,
                         df_all[["word", "most_common", "Noun_binary",
                                 "Verb_binary", "Adj_binary", "Adverb_binary"]],
                         on="word")

    df_causal.to_csv("data/causal_data_input_pos4_binary_min" + str(MIN) + to_str[COMBINED]
                     + pc_to_str[PERCENT] + ".csv")

    #causal_data = pd.read_csv("data/causal_data_input.csv")
    # df_causal = pd.merge(causal_data,
    #                      df_all[["word", "most_common", "Noun_binary", "Verb_binary", "Adj_binary", "Adverb_binary"]],
    #                      on="word")
    # df_causal.to_csv("data/causal_data_input_pos4_binary_min10t.csv")
    # analyse_pos_tags()
