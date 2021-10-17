"""
Turn tweets into word representations
"""
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import regex as re
from tqdm import tqdm
import copy
import pickle
import os
import pandas as pd

import re
import argparse

def text_to_list(text):
    sentence_list = re.split("\n", text)
    return [re.split("\s", sentence) for sentence in sentence_list]

def filter_on_occurrences(input_corpus, target_words):
    """
    only keep sentences that contain the target words
    """
    output_corpus = []
    for sentence in tqdm(input_corpus):
        for target in target_words:
            if target in sentence:
                output_corpus.append(sentence)
    return output_corpus

def prepare_sentence(sent):
    """
    Pre-process a tweet (i.e. sentence):
     - remove urls hashtags & some special charachters
     - separate punctuation from the word so they are tokenized separately
    """
    if type(sent) == str:
        sent = sent.split(" ")
    elif type(sent) == int or type(sent) == float:
        return [""]
    sent = [wrd for wrd in sent if "http" not in wrd]
    sent = [wrd if "@" not in wrd else "user" for wrd in sent]
    sent = " ".join(sent)
    # Change to lower case
    sent = sent.lower()
    # Remove certain tokens
    sent = sent.replace('”', '').replace('“','').replace('‘','')
    # change tabs and new lines into spaces
    sent = sent.replace("\n", " ").replace("\t", " ")
    # Separate punctuation from the word
    for punct in ["?","!",",",".","\"", "\'",";",":",")","]"]:
        sent = sent.replace(punct, " " + punct)
        sent = sent.replace("  "," ")
    # Some punctuation marks may also be before a word
    for punct in [",",".","\"", "\'","(","["]:
        sent = sent.replace(punct, punct + " ")
        sent = sent.replace("  "," ")
    return sent

def tokenize_sentence(sent, tokenizer):
    """
    :param sent: a string (or list of strings) representing the sentence
    tokenizer: a pretrained tokenizer
    :return
    - decoded_list : a tokenized list of words representing the sentence
    - tokens : tokenizer output
    """
    sent = prepare_sentence(sent)
    tokens = tokenizer(sent, return_tensors="pt")

    decoded = tokenizer.decode(tokens["input_ids"][0], clean_up_tokenization_spaces=False)
    decoded = decoded[3:][:-4]
    decoded_list = ["<s>"]
    decoded_list = decoded_list + decoded.split(" ")
    decoded_list.append("</s>")

    return decoded_list, tokens

def get_word_representation(target_word, sentence, model, tokenizer, reps="sum"):
    """
    Takes a target word and its sentence and outputs the word representation for that word,
    as the sum over all the hidden state representations, the first layer representation or
    the last layer representation
    Note: If target_word appears twice in sentence, will only pick the first one
    """
    decoded_list, inputs = tokenize_sentence(sentence, tokenizer)

    if target_word not in decoded_list:
        #print("Oddly enough,",target_word,"is not in",decoded_list)
        return 0
    else:
        idx = decoded_list.index(target_word)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            if reps == "sum":  # sum hidden states
                hidden_states = torch.stack(list(outputs.hidden_states), dim=0)
                return torch.sum(hidden_states, dim=0)[0][idx]
            elif reps == "last":  # take last layer only
                return outputs.hidden_states[-1][0][idx]
            elif reps == "first":  # take first layer only
                return outputs.hidden_states[1][0][idx]

def get_word_sentence_representation(target_word, sentence, model, tokenizer, reps="sum"):
    """
    Takes a target word and its sentence and outputs the word representation for that word,
    as the sum over all the hidden state representations, the first layer representation or
    the last layer representation
    Note: If target_word appears twice in sentence, will only pick the first one
    """

    decoded_list, inputs = tokenize_sentence(sentence, tokenizer)

    if target_word not in decoded_list:
        #print("Oddly enough,",target_word,"is not in",decoded_list)
        return 0, decoded_list
    else:
        idx = decoded_list.index(target_word)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            if reps == "sum":  # sum hidden states
                hidden_states = torch.stack(list(outputs.hidden_states), dim=0)
                return torch.sum(hidden_states, dim=0)[0][idx], decoded_list
            elif reps == "last":  # take last layer only
                return outputs.hidden_states[-1][0][idx], decoded_list
            elif reps == "first":  # take first layer only
                return outputs.hidden_states[1][0][idx], decoded_list

def get_word_representations_for_corpus(corpus, targets, model, tokenizer, reps, is_semeval=True):

    def remove_pos(word, sent):
        if is_semeval:
            idx = sent.index(word)
            sent[idx] = sent[idx][:-3]
            word = word[:-3]
        return word, sent

    corpus_reps, sentence_reps = {}, {}
    for sentence in tqdm(corpus):
        for target in targets:
            if target in sentence:
                # make sure to remove POS tag from target word and sentence
                target_word = copy.deepcopy(target)
                context_sentence = copy.deepcopy(sentence)
                target_word, context_sentence = remove_pos(target_word, context_sentence)
                word_rep, sent_rep = get_word_sentence_representation(target_word,
                                                                      context_sentence,
                                                                      model,
                                                                      tokenizer,
                                                                      reps=reps)
                if type(word_rep) != int:
                    corpus_reps[target] = corpus_reps[target] + [word_rep] if target in corpus_reps.keys() else [word_rep]
                    sentence_reps[target] = sentence_reps[target] + [word_rep] if target in sentence_reps.keys() else [word_rep]
    return corpus_reps, sentence_reps

def sentences2representations(target, sentences, model, tokenizer, reps):
    """
    Tokenize sentences and get the word representations
    """
    target = target.lower()
    representations, decoded_sentences = [], []
    for sent in sentences:
        context_sentence = copy.deepcopy(sent)
        word_rep, decoded_sent = get_word_sentence_representation(
                                           target_word=target,
                                           sentence=context_sentence,
                                           model=model,
                                           tokenizer=tokenizer,
                                           reps=reps)
        # Check if word representation is valid
        if type(word_rep) != int:
            representations.append(word_rep)
            decoded_sentences.append(decoded_sent)
    return  representations, decoded_sentences

def tweets2representatons(targets, model, tokenizer, reps="sum", data_path="slang_word_tweets"):
    repr_dict = {}
    text_dict = {}
    for word in tqdm(targets):
        word_df_path = os.path.join(data_path, "tweets_df_"+str(word)+".csv")
        try:
            word_df = pd.read_csv(word_df_path)
        except: #FileNotFoundError:
            # no data for this word, so skip it
            continue
        sentences = word_df.text
        representations, decoded_sentences = \
            sentences2representations(target=word,
                                      sentences=sentences,
                                      model=model,
                                      tokenizer=tokenizer,
                                      reps=reps)
        repr_dict[word] = representations
        text_dict[word] = decoded_sentences
        print("for "+  str(word) + ": kept", len(decoded_sentences), "tweets out of", len(sentences))
    return repr_dict, text_dict

def get_tweet_reprs(words_list, reps="sum", data_path="slang_word_tweets",
                    model_path="models/roberta_UD_5e-05"):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    repr_dict, text_dict = tweets2representatons(targets=words_list,
                                                 model=model,
                                                 tokenizer=tokenizer,
                                                 data_path=data_path,
                                                 reps=reps
                                                 )
    return repr_dict, text_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-eval", type=bool, default=False)
    parser.add_argument("--reps", type=str, default="sum")
    parser.add_argument("--data-path", type=str, default='data/tweets_new/hybrid_word_tweets')
    parser.add_argument("--type", type=str, default="slang") #["slang","nonslang","both",[CUSTOM_LIST]]
    parser.add_argument("--semeval-path", type=str, default='data/semeval2020_ulscd_eng')
    parser.add_argument("--model-path",type=str,default="models/roberta_UD")
    args = parser.parse_args()

    if args.sem_eval:
        corpus1_sents_lemmatized = open(os.path.join(args.semeval_path,"corpus1/lemma/ccoha1.txt")).read().strip()
        corpus2_sents_lemmatized = open(os.path.join(args.semeval_path,"corpus2/lemma/ccoha2.txt")).read().strip()
        target_words = open(os.path.join(args.semeval_path,"targets.txt")).read().strip()

        corpus1_lemma = text_to_list(corpus1_sents_lemmatized)
        corpus2_lemma = text_to_list(corpus2_sents_lemmatized)
        target_words = [word for word in re.split("\n", target_words)]

        corpus1 = filter_on_occurrences(corpus1_lemma, target_words)
        corpus2 = filter_on_occurrences(corpus2_lemma, target_words)

        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')

        corpus1_reps, sentence1_reps = get_word_representations_for_corpus(corpus1, target_words, model, tokenizer,
                                                                           reps=args.reps)
        corpus2_reps,  sentence2_reps = get_word_representations_for_corpus(corpus2, target_words, model, tokenizer,
                                                                            reps=args.reps)

        with open('corpus1_'+args.reps+'_layer_reps.pickle', 'wb') as handle:
            pickle.dump(corpus1_reps, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('corpus2_'+args.reps+'_layer_reps.pickle', 'wb') as handle:
            pickle.dump(corpus2_reps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        data_path = args.data_path

        if args.type in ["slang","nonslang","both"]:

            words_path = "data/word-lists/all_words_300.csv"
            selected_words_df = pd.read_csv(words_path)
            words_list = list(selected_words_df[selected_words_df.type == args.type].word)

            counter = 0
            for word in words_list:
                word_df_path = os.path.join(data_path, "tweets_df_" + str(word) + ".csv")
                print(word_df_path)
                try:
                    word_df = pd.read_csv(word_df_path, lineterminator='\n')
                    if len(word_df.word) < 200:
                        print(word, "only has", len(word_df.word), "tweets")
                    else:
                        counter += 1
                except:
                    print(word, "is not working")

            print("There are", counter, "complete files")

            repr_dict, text_dict = get_tweet_reprs(words_list=words_list,
                                                   reps=args.reps, data_path=args.data_path)

            with open("data/" + data_path.split("/")[1].split("_")[1] + f'_{args.type}_reps.pickle', 'wb') as handle:
                pickle.dump(repr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open("data/" + data_path.split("/")[1].split("_")[1] + f'_{args.type}_tweets.pickle', 'wb') as handle:
                pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            repr_dict, text_dict = get_tweet_reprs(words_list=args.type,
                                                   reps=args.reps, data_path=args.data_path)

            with open("data/" + data_path.split("/")[1].split("_")[1] + f'_{args.type}_reps.pickle', 'wb') as handle:
                pickle.dump(repr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open("data/" + data_path.split("/")[1].split("_")[1] + f'_{args.type}_tweets.pickle', 'wb') as handle:
                pickle.dump(text_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
