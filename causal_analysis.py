from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from mlxtend.evaluate import permutation_test
import copy
import random
from visualizations import plot_log_freqs_change

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

def other_perm_test(x, y):
    gT = np.abs(np.average(x) - np.average(y))
    #Pool variables into one single distribution:
    pV = list(x) + list(y)
    # Copy pooled distribution:
    pS = copy.copy(pV)
    # Initialize permutation:
    pD = []
    # Define p (number of permutations):
    p = 1000
    # Permutation loop:
    for i in range(0, p):
        # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled distributions and store it in pD:
        pD.append(np.abs(np.average(pS[0:int(len(pS) / 2)]) - np.average(pS[int(len(pS) / 2):])))
    p_val = len(np.where(pD >= gT)[0]) / p
    return p_val

def independence_tests(slang_scores, nonslang_scores):
    t_test_statistic, t_test_pval = ttest_ind(slang_scores,nonslang_scores)
    perm_test_pval = permutation_test(slang_scores, nonslang_scores,
                                      method="approximate", seed=111, num_rounds=10000
                                      )
    print("t-test p-value is", t_test_pval, "and permutation test p-value is", perm_test_pval)

if __name__ == '__main__':

    polysemy_file_paths = {"slang": "word-lists/polysemy_slang.csv",
                           "nonslang": "word-lists/polysemy_nonslang.csv",
                           "hybrid": "word-lists/polysemy_hybrid.csv",
                           }

    freq_file_paths = {"slang": "data/frequencies/freq_slang_counts_24h_2010.csv",
                       "nonslang": "data/frequencies/freq_nonslang_counts_24h_2010.csv"
                       }

    causal_df = pd.read_csv("word-lists/causal_data_input.csv")

    hybrid_polysemy_df = pd.read_csv(polysemy_file_paths["hybrid"])
    hybrid_polysemy_df["polysemy"] = hybrid_polysemy_df["num_s"] +  hybrid_polysemy_df["num_ns"]
    hybrid_polysemy_df.to_csv(polysemy_file_paths["hybrid"])

    slang_freq_df = pd.read_csv(freq_file_paths["slang"])
    nonslang_freq_df = pd.read_csv(freq_file_paths["nonslang"])

    plot_log_freqs_change(slang_freq_df=slang_freq_df, nonslang_freq_df=nonslang_freq_df)

    slang_polysemy_df = pd.read_csv(polysemy_file_paths["slang"])
    nonslang_polysemy_df = pd.read_csv(polysemy_file_paths["nonslang"])

    independence_tests(slang_polysemy_df.polysemy.values, nonslang_polysemy_df.polysemy.values)

    plt.hist([slang_polysemy_df.polysemy,nonslang_polysemy_df.polysemy, hybrid_polysemy_df.polysemy],
             color=["orange", "dodgerblue", "orchid"],
             label=["slang", "nonslang", "hybrid"])
    plt.legend()
    plt.title("Number of Word Senses - Distribution")
    plt.xlabel("# word senses")
    plt.show()