import wordfreq
import pandas as pd
import numpy as np
EPSILON = 1e-8

def get_close_words(word, freq_dict, epsilon=1e-100, minimum=0.0):
    freq = wordfreq.word_frequency(word, lang='en', wordlist='best', minimum=minimum)
    close_words = [k for k,v in freq_dict.items() if abs(v-freq)<epsilon]
    return close_words

slang_words = pd.read_csv("selected_words.csv")
slang_words_list = slang_words.selected_words
en_freq_dict = wordfreq.get_frequency_dict('en', wordlist='best')
frequencies = sorted(list(en_freq_dict.values()))
MIN_FREQ = min(frequencies)

nonslang = pd.read_csv("selected_nonslang_words.csv")
nonslang_words, nonslang_words2, nonslang_words3, nonslang_words4, = [], [], [], []

for word in slang_words_list:
    print("now looking at", word)
    freq = wordfreq.word_frequency(word, lang='en', wordlist='best', minimum=MIN_FREQ)
    print("frequency of", word, "=", freq)
    if freq == MIN_FREQ:
        close_words = get_close_words(word=word, freq_dict=en_freq_dict, epsilon=1e-80, minimum=MIN_FREQ)
    else:
        close_words = get_close_words(word=word, freq_dict=en_freq_dict, epsilon=EPSILON, minimum=MIN_FREQ)
    print("got", len(close_words), "close words for", word)
    eps = EPSILON
    while len(close_words) == 0:
        eps = eps*100
        close_words = get_close_words(word=word, freq_dict=en_freq_dict, epsilon=eps, minimum=MIN_FREQ)
    random_word = np.random.choice(close_words)
    nonslang_words.append(random_word)

    random_word2 = np.random.choice(close_words)
    nonslang_words2.append(random_word2)

    random_word3 = np.random.choice(close_words)
    nonslang_words3.append(random_word3)

    random_word4 = np.random.choice(close_words)
    nonslang_words4.append(random_word4)
    print("chose", random_word, random_word2, random_word3, random_word4)

print("nonslang words", nonslang_words)

nonslang["nonslang5"] = nonslang_words
nonslang["nonslang6"] = nonslang_words2
nonslang["nonslang7"] = nonslang_words3
nonslang["nonslang8"] = nonslang_words4

nonslang.to_csv("selected_nonslang_words.csv")


