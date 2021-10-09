from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from mlxtend.evaluate import permutation_test


def independence_tests(slang_scores, nonslang_scores):
    t_test_statistic, t_test_pval = ttest_ind(slang_scores,nonslang_scores)
    perm_test_pval = permutation_test(slang_scores,nonslang_scores,
                                      method="approximate", seed=111, num_rounds=10000
                                      )
    print("t-test p-value is", t_test_pval, "and permutation test p-value is", perm_test_pval)

polysemy_file_paths = {"slang": "word-lists/polysemy_slang.csv",
                       "nonslang":"word-lists/polysemy_nonslang.csv",
                       "hybrid": "word-lists/polysemy_hybrid.csv",
                       }

freq_file_paths = {"slang": "data/frequencies/freq_slang_counts_24h.csv",
                    "nonslang":"data/frequencies/freq_nonslang_counts_24h.csv"
                       }

hybrid_polysemy_df = pd.read_csv(polysemy_file_paths["hybrid"])
hybrid_polysemy_df["polysemy"] = hybrid_polysemy_df["num_s"] +  hybrid_polysemy_df["num_ns"]
hybrid_polysemy_df.to_csv(polysemy_file_paths["hybrid"])

slang_freq_df = pd.read_csv(freq_file_paths["slang"])
nonslang_freq_df = pd.read_csv(freq_file_paths["nonslang"])

plt.hist([[np.log(k) for k in slang_freq_df.freq.values],
          [np.log(k) for k in nonslang_freq_df.freq.values]],
         color=["orange","dodgerblue"],
         label=["slang","nonslang"])
plt.xlabel("log of # occurrences in 24 hours")
plt.legend()
plt.title("log-frequency of words in 2010")
plt.show()

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