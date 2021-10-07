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
                       "nonslang":"word-lists/polysemy_nonslang.csv"
                       }

freq_file_paths = {"slang": "data/frequencies/freq_slang.csv",
                    "nonslang":"data/frequencies/freq_nonslang.csv"
                       }

slang_freq_df = pd.read_csv(freq_file_paths["slang"])
plt.hist(slang_freq_df.freq.values, color="orange")
plt.xlabel("# occurrences in 48 hours")
plt.title("frequency of slang words in 2010")
plt.show()

slang_polysemy_df = pd.read_csv(polysemy_file_paths["slang"])
nonslang_polysemy_df = pd.read_csv(polysemy_file_paths["nonslang"])

slang_polysemy_scores = slang_polysemy_df.polysemy.values
nonslang_polysemy_scores = nonslang_polysemy_df.polysemy.values

independence_tests(slang_polysemy_scores, nonslang_polysemy_scores)
plt.hist([slang_polysemy_df.polysemy,nonslang_polysemy_df.polysemy],
         color=["orange", "dodgerblue"],
         label=["slang", "nonslang"])
plt.legend()
plt.title("Number of Word Senses - Distribution")
plt.xlabel("# word senses")
plt.show()