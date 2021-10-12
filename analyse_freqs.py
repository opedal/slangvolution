import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from causal_analysis import independence_tests

file_names = {"slang 2020": 'data/frequencies/freq_slang_counts_24h_2020.csv',
              "slang 2010": 'data/frequencies/freq_slang_counts_24h_2010.csv',
              "nonslang 2020": 'data/frequencies/freq_nonslang_counts_24h_2020.csv',
              "nonslang 2010": 'data/frequencies/freq_nonslang_counts_24h_2010.csv',
              "sample 2020": 'data/frequencies/freq_sample_words_24h_2020.csv',
              "sample 2010": 'data/frequencies/freq_sample_words_24h_2010.csv',
              }

def divide_by_larger_freq(X, colname1="freq2020", colname2="freq2010"):
    x,y = X[colname1], X[colname2]
    if x >= y:
        return x/y
    else:
        return y/x

def merge_freq_dfs(freq2010, freq2020, NORMALIZING_CONSTANT=6.4):
    freq2010.columns = ['freq2010', 'word', 'year', 'type']
    freq2020.columns = ['freq2020', 'word', 'year', 'type']
    all_freqs = pd.merge(freq2010, freq2020[['freq2020', 'word']], on="word", how="inner")
    all_freqs["freq2020_norm"] = all_freqs.freq2020.apply(lambda x: x / NORMALIZING_CONSTANT)
    all_freqs["freq_diff"] = all_freqs.freq2020 / (all_freqs.freq2010)
    all_freqs["freq_diff_norm"] = all_freqs.freq2020_norm / (all_freqs.freq2010)

    all_freqs["div_freq_larger"] = all_freqs[["freq2020", "freq2010"]].apply(divide_by_larger_freq, axis=1)
    all_freqs["div_freq_larger_norm"] = all_freqs[["freq2020_norm", "freq2010"]].apply(
                                      lambda x: divide_by_larger_freq(x, colname1="freq2020_norm"), axis=1)
    #all_freqs["abs_diff"] = np.abs(all_freqs.freq2020 - all_freqs.freq2010)
    #all_freqs["abs_diff_norm"] = all_freqs["abs_diff"] / (all_freqs.freq2010)

    all_freqs["abs_diff"] = np.abs(all_freqs.freq2020_norm - all_freqs.freq2010)
    all_freqs["abs_diff_norm"] = all_freqs["abs_diff"] / (all_freqs.freq2010)

    all_freqs["abs_diff_norm2020"] = all_freqs["abs_diff"] / (all_freqs.freq2020_norm)

    all_freqs["relative_diff"] = 2*np.abs(all_freqs.freq2020_norm - all_freqs.freq2010)/ \
                                 (all_freqs.freq2020_norm + all_freqs.freq2010)

    all_freqs["log_diff"] = np.log(all_freqs.freq2020_norm/all_freqs.freq2010)

    return all_freqs

def freq_difference():
    slang2010 = pd.read_csv(file_names["slang 2010"])
    slang2020 = pd.read_csv(file_names["slang 2020"])
    s_all = merge_freq_dfs(slang2010, slang2020)
    s_all.to_csv("data/frequencies/slang_all_freqs.csv")
    s_all = s_all[s_all.freq_diff < 10000]

    ns2010 = pd.read_csv(file_names["nonslang 2010"])
    ns2020 = pd.read_csv(file_names["nonslang 2020"])
    ns_all = merge_freq_dfs(ns2010, ns2020)
    ns_all.to_csv("data/frequencies/nonslang_all_freqs.csv")

    sample2010 = pd.read_csv(file_names["sample 2010"])
    sample2020 = pd.read_csv(file_names["sample 2020"])
    sample_all = merge_freq_dfs(sample2010, sample2020)

    plt.hist([ns_all["div_freq_larger_norm"][ns_all["div_freq_larger_norm"] < 200], s_all["div_freq_larger_norm"][s_all["div_freq_larger_norm"] < 200]], label = ["nonslang", "slang"], bins=22)

def plot_slang_nonslang_comparison(s_all, ns_all, curr_col="abs_diff2_norm",
                                   title="absolute subtracted difference",
                                   bins=22):
    plt.hist([ns_all[curr_col], s_all[curr_col]],
             label=["nonslang", "slang"],
             color=["mediumslateblue", "darkorange"],
             bins=bins)
    plt.legend()
    plt.title(title+" between 2010 and 2020 frequencies \n slang versus nonslang")
    plt.show()

def print_slang_nonslang_comparison(s_all, ns_all):
    print("results for normalized absolute distance using normalized 2020:")
    print_averages_and_medians(s_all=s_all, ns_all=ns_all, curr_col = "abs_diff2_norm")

    print("results for normalized absolute distance using original 2020:")
    print_averages_and_medians(s_all=s_all, ns_all=ns_all, curr_col = "abs_diff_norm")

    print("results for normalized 2020 freqs divided by 2010 freqs:")
    print_averages_and_medians(s_all=s_all, ns_all=ns_all, curr_col="freq_diff_norm")

    print("results for larger year freqs divided by smaller freqs, with normalized 2020:")
    print_averages_and_medians(s_all=s_all, ns_all=ns_all, curr_col="div_freq_larger_norm")

def print_averages_and_medians(s_all, ns_all, curr_col):
    print("average: slang={}, nonslang={}, \n median: slang={}, nonslang={}".format(
                                                        "%.2f" % np.average(s_all[curr_col]),
                                                        "%.2f" % np.average(ns_all[curr_col]),
                                                        "%.2f" % np.median(s_all[curr_col]),
                                                        "%.2f" % np.median(ns_all[curr_col]),
                                                                        ))

def average_frequency():
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
    print("the frequency of sample words between 2010 and 2020, increased times ",
          avgs["sample 2020"]/avgs["sample 2010"])


average_frequency()
freq_difference()

