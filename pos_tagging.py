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

def prep_for_causal():
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
        df_slang[pos] = df_slang[pos + "_x"] + df_slang[pos + "_y"]
    df_slang['most_common'] = df_slang[POS].idxmax(axis=1)

    df_nonslang = pd.merge(df_new_nonslang, df_old_nonslang, on="word")
    for pos in POS:
        df_nonslang[pos] = df_nonslang[pos + "_x"] + df_nonslang[pos + "_y"]
    df_nonslang['most_common'] = df_nonslang[POS].idxmax(axis=1)
    df_all = pd.concat([df_slang, df_nonslang])

    MIN = 5
    for pos in POS:
        df_all[pos+"_binary"] = df_all[pos].apply(lambda x: int(x > MIN))
    return df_all

if __name__ == '__main__':
    df_all = prep_for_causal()
    causal_data = pd.read_csv("data/causal_data_input.csv")
    df_causal = pd.merge(causal_data,
                         df_all[["word", "most_common", "Noun_binary", "Verb_binary", "Adj_binary", "Adverb_binary"]],
                         on="word")
    df_causal.to_csv("data/causal_data_input_pos4_binary.csv")
    analyse_pos_tags()
